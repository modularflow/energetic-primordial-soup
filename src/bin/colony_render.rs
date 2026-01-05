//! Offline colony visualizer for raw soup dumps
//!
//! Reads `raw_epoch_*.bin` files (BFFR/BFF2 formats) and identifies the top-K
//! most common program tapes in a chosen snapshot (default: last epoch).
//! Then renders per-epoch PNGs showing only those colonies (as a one-pixel-per-program map).
//!
//! Usage (example):
//!   cargo run --release --bin colony_render -- \
//!     --raw-dir raw_data --out-dir colony_viz --top-k 10 --exclude-zero
//!
//! Then turn frames into videos:
//!   ffmpeg -framerate 30 -i colony_viz/overlay/%08d.png -pix_fmt yuv420p overlay.mp4
//!   ffmpeg -framerate 30 -i colony_viz/colony_00/%08d.png -pix_fmt yuv420p colony_00.mp4

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const SINGLE_TAPE_SIZE: usize = 64;

#[derive(Debug, Clone)]
struct RawDataHeader {
    epoch: usize,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    layout: [usize; 2],
}

fn usage_and_exit(msg: Option<&str>) -> ! {
    if let Some(m) = msg {
        eprintln!("Error: {m}\n");
    }
    eprintln!("colony_render (offline raw dump visualizer)");
    eprintln!();
    eprintln!("REQUIRED:");
    eprintln!("  --raw-dir <DIR>       Directory containing raw_epoch_*.bin");
    eprintln!("  --out-dir <DIR>       Output directory for rendered frames + CSV");
    eprintln!();
    eprintln!("OPTIONAL:");
    eprintln!("  --top-k <N>           Number of colonies to track (default: 10)");
    eprintln!("  --pick-epoch <EPOCH>  Choose snapshot epoch to select top-K (default: last file)");
    eprintln!("  --exclude-zero        Exclude all-zero tapes from colony selection");
    eprintln!();
    std::process::exit(2);
}

fn parse_args() -> (String, String, usize, Option<usize>, bool) {
    let argv: Vec<String> = std::env::args().collect();
    let mut raw_dir: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut top_k: usize = 10;
    let mut pick_epoch: Option<usize> = None;
    let mut exclude_zero = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--raw-dir" => {
                i += 1;
                raw_dir = argv.get(i).cloned();
            }
            "--out-dir" => {
                i += 1;
                out_dir = argv.get(i).cloned();
            }
            "--top-k" => {
                i += 1;
                let v = argv.get(i).ok_or("--top-k requires a value").ok();
                top_k = v
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| usage_and_exit(Some("Invalid --top-k value")));
                if top_k == 0 {
                    usage_and_exit(Some("--top-k must be > 0"));
                }
            }
            "--pick-epoch" => {
                i += 1;
                let v = argv.get(i).ok_or("--pick-epoch requires a value").ok();
                pick_epoch = Some(
                    v.and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or_else(|| usage_and_exit(Some("Invalid --pick-epoch value"))),
                );
            }
            "--exclude-zero" => {
                exclude_zero = true;
            }
            "--help" | "-h" => usage_and_exit(None),
            other => usage_and_exit(Some(&format!("Unknown argument: {other}"))),
        }
        i += 1;
    }

    let raw_dir = raw_dir.unwrap_or_else(|| usage_and_exit(Some("--raw-dir is required")));
    let out_dir = out_dir.unwrap_or_else(|| usage_and_exit(Some("--out-dir is required")));
    (raw_dir, out_dir, top_k, pick_epoch, exclude_zero)
}

fn list_raw_files(raw_dir: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(raw_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "bin"))
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |s| s.starts_with("raw_epoch_"))
        })
        .collect();

    files.sort();
    Ok(files)
}

/// Load raw data file (header + soup data) - supports both v1 BFFR and v2 BFF2 formats
fn load_raw_data(path: &Path) -> std::io::Result<(RawDataHeader, Vec<u8>)> {
    let mut file = BufReader::new(File::open(path)?);

    let mut header_buf = [0u8; 32];
    file.read_exact(&mut header_buf)?;

    let magic = &header_buf[0..4];
    let is_compressed = magic == b"BFF2";
    if magic != b"BFFR" && magic != b"BFF2" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid raw data magic (expected BFFR or BFF2)",
        ));
    }

    let epoch = u32::from_le_bytes(header_buf[8..12].try_into().unwrap()) as usize;
    let grid_width = u32::from_le_bytes(header_buf[12..16].try_into().unwrap()) as usize;
    let grid_height = u32::from_le_bytes(header_buf[16..20].try_into().unwrap()) as usize;
    let num_sims = u32::from_le_bytes(header_buf[20..24].try_into().unwrap()) as usize;
    let layout_cols = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;
    let layout_rows = u32::from_le_bytes(header_buf[28..32].try_into().unwrap()) as usize;

    let header = RawDataHeader {
        epoch,
        grid_width,
        grid_height,
        num_sims,
        layout: [layout_cols, layout_rows],
    };

    let soup_data = if is_compressed {
        let mut size_buf = [0u8; 4];
        file.read_exact(&mut size_buf)?;
        let compressed_size = u32::from_le_bytes(size_buf) as usize;
        let mut compressed = vec![0u8; compressed_size];
        file.read_exact(&mut compressed)?;
        zstd::decode_all(&compressed[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
    } else {
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        data
    };

    Ok((header, soup_data))
}

/// Fast, stable 64-bit hash (FNV-1a)
fn hash64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

fn pick_top_k_hashes(
    soup: &[u8],
    header: &RawDataHeader,
    top_k: usize,
    exclude_zero: bool,
) -> Vec<(u64, usize)> {
    let programs_per_sim = header.grid_width * header.grid_height;
    let total_programs = programs_per_sim * header.num_sims;
    let expected_len = total_programs * SINGLE_TAPE_SIZE;
    if soup.len() < expected_len {
        eprintln!(
            "Warning: soup length {} < expected {}, proceeding with min length",
            soup.len(),
            expected_len
        );
    }

    let mut counts: HashMap<u64, usize> = HashMap::new();
    for prog_idx in 0..total_programs {
        let start = prog_idx * SINGLE_TAPE_SIZE;
        let end = start + SINGLE_TAPE_SIZE;
        if end > soup.len() {
            break;
        }
        let tape = &soup[start..end];
        if exclude_zero && is_all_zero(tape) {
            continue;
        }
        let h = hash64(tape);
        *counts.entry(h).or_insert(0) += 1;
    }

    let mut vec: Vec<(u64, usize)> = counts.into_iter().collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1)); // desc by count
    vec.truncate(top_k);
    vec
}

fn colony_palette(k: usize) -> Vec<[u8; 3]> {
    // Simple distinct-ish palette (repeats if k > len).
    let base: &[[u8; 3]] = &[
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 128, 255],
        [255, 165, 0],
        [200, 0, 200],
        [0, 200, 200],
        [255, 255, 0],
        [128, 0, 255],
        [0, 255, 128],
        [255, 0, 128],
        [128, 255, 0],
    ];
    (0..k).map(|i| base[i % base.len()]).collect()
}

fn save_png_rgb(path: &Path, width: usize, height: usize, img: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder
        .write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer
        .write_image_data(img)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let (raw_dir, out_dir, top_k, pick_epoch, exclude_zero) = parse_args();
    let files = list_raw_files(&raw_dir)?;
    if files.is_empty() {
        usage_and_exit(Some("No raw_epoch_*.bin files found in --raw-dir"));
    }

    // Pick snapshot file to choose colonies from.
    let (pick_path, pick_epoch_label) = if let Some(epoch) = pick_epoch {
        let wanted = format!("raw_epoch_{:08}.bin", epoch);
        let p = files
            .iter()
            .find(|p| p.file_name().and_then(|s| s.to_str()) == Some(wanted.as_str()))
            .cloned()
            .unwrap_or_else(|| usage_and_exit(Some("Requested --pick-epoch file not found")));
        (p, epoch)
    } else {
        let p = files.last().unwrap().clone();
        // best-effort epoch label from filename
        let epoch = p
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("raw_epoch_"))
            .and_then(|s| s.strip_suffix(".bin"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        (p, epoch)
    };

    println!(
        "Found {} raw files. Selecting colonies from epoch {} ({})",
        files.len(),
        pick_epoch_label,
        pick_path.display()
    );

    let (pick_header, pick_soup) = load_raw_data(&pick_path)?;
    let top = pick_top_k_hashes(&pick_soup, &pick_header, top_k, exclude_zero);
    if top.is_empty() {
        usage_and_exit(Some("No colonies found (maybe everything is zero and --exclude-zero was set?)"));
    }

    // Prepare output dirs
    fs::create_dir_all(&out_dir)?;
    let overlay_dir = Path::new(&out_dir).join("overlay");
    fs::create_dir_all(&overlay_dir)?;
    let mut colony_dirs = Vec::new();
    for i in 0..top.len() {
        let d = Path::new(&out_dir).join(format!("colony_{:02}", i));
        fs::create_dir_all(&d)?;
        colony_dirs.push(d);
    }

    // Write selection CSV
    let selection_csv = Path::new(&out_dir).join("colonies_selected.csv");
    {
        let mut f = std::io::BufWriter::new(File::create(&selection_csv)?);
        use std::io::Write;
        writeln!(f, "rank,hash64,count_in_pick_epoch,pick_epoch")?;
        for (i, (h, c)) in top.iter().enumerate() {
            writeln!(f, "{},{:016x},{},{}", i, h, c, pick_header.epoch)?;
        }
    }

    // Write per-epoch counts CSV and render frames
    let counts_csv = Path::new(&out_dir).join("colonies_counts.csv");
    let colors = colony_palette(top.len());
    let mut counts_out = std::io::BufWriter::new(File::create(&counts_csv)?);
    {
        use std::io::Write;
        write!(counts_out, "epoch")?;
        for i in 0..top.len() {
            write!(counts_out, ",colony_{:02}_count", i)?;
        }
        writeln!(counts_out)?;
    }

    for (file_idx, path) in files.iter().enumerate() {
        let (header, soup) = match load_raw_data(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping {} (load error: {})", path.display(), e);
                continue;
            }
        };

        let programs_per_sim = header.grid_width * header.grid_height;
        let total_programs = programs_per_sim * header.num_sims;
        let expected = total_programs * SINGLE_TAPE_SIZE;
        if soup.len() < expected {
            eprintln!(
                "Warning: {} soup len {} < expected {}",
                path.display(),
                soup.len(),
                expected
            );
        }

        // Render one pixel per program in mega layout.
        let mega_w = header.grid_width * header.layout[0];
        let mega_h = header.grid_height * header.layout[1];
        let mut overlay = vec![0u8; mega_w * mega_h * 3];
        let mut per_colony_imgs: Vec<Vec<u8>> = (0..top.len())
            .map(|_| vec![0u8; mega_w * mega_h * 3])
            .collect();
        let mut counts = vec![0usize; top.len()];

        // Map from hash -> colony index (top-k only)
        let mut colony_index: HashMap<u64, usize> = HashMap::new();
        for (i, (h, _)) in top.iter().enumerate() {
            colony_index.insert(*h, i);
        }

        for abs_prog_idx in 0..total_programs {
            let start = abs_prog_idx * SINGLE_TAPE_SIZE;
            let end = start + SINGLE_TAPE_SIZE;
            if end > soup.len() {
                break;
            }
            let tape = &soup[start..end];
            let h = hash64(tape);
            let Some(&ci) = colony_index.get(&h) else { continue };

            counts[ci] += 1;

            // Convert abs index to mega-grid (sim tile + local x/y).
            let sim_idx = abs_prog_idx / programs_per_sim;
            let local = abs_prog_idx % programs_per_sim;
            let lx = local % header.grid_width;
            let ly = local / header.grid_width;
            let sim_col = sim_idx % header.layout[0];
            let sim_row = sim_idx / header.layout[0];
            if sim_row >= header.layout[1] {
                continue;
            }
            let gx = sim_col * header.grid_width + lx;
            let gy = sim_row * header.grid_height + ly;
            let pix = (gy * mega_w + gx) * 3;
            let col = colors[ci];

            // Overlay shows all colonies, last one wins on collisions (shouldn't happen unless hash collision).
            overlay[pix] = col[0];
            overlay[pix + 1] = col[1];
            overlay[pix + 2] = col[2];

            // Per-colony mask
            per_colony_imgs[ci][pix] = col[0];
            per_colony_imgs[ci][pix + 1] = col[1];
            per_colony_imgs[ci][pix + 2] = col[2];
        }

        // Write counts row
        {
            use std::io::Write;
            write!(counts_out, "{}", header.epoch)?;
            for c in &counts {
                write!(counts_out, ",{}", c)?;
            }
            writeln!(counts_out)?;
        }

        // Save images
        let overlay_path = overlay_dir.join(format!("{:08}.png", header.epoch));
        save_png_rgb(&overlay_path, mega_w, mega_h, &overlay)?;
        for (i, img) in per_colony_imgs.iter().enumerate() {
            let p = colony_dirs[i].join(format!("{:08}.png", header.epoch));
            save_png_rgb(&p, mega_w, mega_h, img)?;
        }

        if file_idx % 10 == 0 {
            println!(
                "Rendered {}/{} (epoch {})",
                file_idx + 1,
                files.len(),
                header.epoch
            );
        }
    }

    println!("Done.");
    println!("Selection CSV: {}", selection_csv.display());
    println!("Counts CSV: {}", counts_csv.display());
    println!("Overlay frames: {}/overlay/%08d.png", out_dir);
    println!("Per-colony frames: {}/colony_XX/%08d.png", out_dir);

    Ok(())
}


