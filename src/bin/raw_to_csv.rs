//! Export raw soup dumps to CSV (offline).
//!
//! WARNING: This can get huge fast. Rows = (#raw files) * (num_sims * grid_w * grid_h).
//! For 16 sims * 128*128 = 262,144 rows per epoch. If you saved 1,500 epochs, that's ~393M rows.
//!
//! Typical usage:
//!   # Export a single epoch (recommended)
//!   cargo run --release --bin raw_to_csv -- --raw-dir raw_data --pick-epoch 99968 --out soup_99968.csv
//!
//!   # Export every 10th raw file
//!   cargo run --release --bin raw_to_csv -- --raw-dir raw_data --stride 10 --out soup_stride10.csv
//!
//! Output columns:
//!   epoch,sim_idx,x,y,hash64,is_zero[,bytes_hex]
//!
//! Use `--bytes-hex N` to include the first N bytes of each 64-byte tape in hex.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const SINGLE_TAPE_SIZE: usize = 64;

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    eprintln!("raw_to_csv (offline raw dump exporter)");
    eprintln!();
    eprintln!("REQUIRED:");
    eprintln!("  --raw-dir <DIR>         Directory containing raw_epoch_*.bin");
    eprintln!("  --out <FILE.csv>        Output CSV file");
    eprintln!();
    eprintln!("OPTIONAL:");
    eprintln!("  --pick-epoch <EPOCH>    Export only this epoch (fastest / safest)");
    eprintln!("  --stride <N>            Export every Nth raw file (default: 1)");
    eprintln!("  --bytes-hex <N>         Include first N bytes of each tape as hex (default: 0)");
    eprintln!("  --exclude-zero          Skip all-zero tapes");
    eprintln!();
    std::process::exit(2);
}

fn parse_args() -> (String, String, Option<usize>, usize, usize, bool) {
    let argv: Vec<String> = std::env::args().collect();
    let mut raw_dir: Option<String> = None;
    let mut out: Option<String> = None;
    let mut pick_epoch: Option<usize> = None;
    let mut stride: usize = 1;
    let mut bytes_hex: usize = 0;
    let mut exclude_zero = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--raw-dir" => {
                i += 1;
                raw_dir = argv.get(i).cloned();
            }
            "--out" => {
                i += 1;
                out = argv.get(i).cloned();
            }
            "--pick-epoch" => {
                i += 1;
                let v = argv.get(i).ok_or("--pick-epoch requires a value").ok();
                pick_epoch = Some(
                    v.and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or_else(|| usage_and_exit(Some("Invalid --pick-epoch value"))),
                );
            }
            "--stride" => {
                i += 1;
                let v = argv.get(i).ok_or("--stride requires a value").ok();
                stride = v
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| usage_and_exit(Some("Invalid --stride value")));
                if stride == 0 {
                    usage_and_exit(Some("--stride must be > 0"));
                }
            }
            "--bytes-hex" => {
                i += 1;
                let v = argv.get(i).ok_or("--bytes-hex requires a value").ok();
                bytes_hex = v
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| usage_and_exit(Some("Invalid --bytes-hex value")));
                bytes_hex = bytes_hex.min(SINGLE_TAPE_SIZE);
            }
            "--exclude-zero" => exclude_zero = true,
            "--help" | "-h" => usage_and_exit(None),
            other => usage_and_exit(Some(&format!("Unknown argument: {other}"))),
        }
        i += 1;
    }

    let raw_dir = raw_dir.unwrap_or_else(|| usage_and_exit(Some("--raw-dir is required")));
    let out = out.unwrap_or_else(|| usage_and_exit(Some("--out is required")));
    (raw_dir, out, pick_epoch, stride, bytes_hex, exclude_zero)
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

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xF) as usize] as char);
    }
    s
}

fn main() -> std::io::Result<()> {
    let (raw_dir, out_path, pick_epoch, stride, bytes_hex, exclude_zero) = parse_args();
    let mut files = list_raw_files(&raw_dir)?;
    if files.is_empty() {
        usage_and_exit(Some("No raw_epoch_*.bin files found in --raw-dir"));
    }

    if let Some(epoch) = pick_epoch {
        let wanted = format!("raw_epoch_{:08}.bin", epoch);
        files = files
            .into_iter()
            .filter(|p| p.file_name().and_then(|s| s.to_str()) == Some(wanted.as_str()))
            .collect();
        if files.is_empty() {
            usage_and_exit(Some("Requested --pick-epoch file not found"));
        }
    }

    let file = File::create(&out_path)?;
    let mut w = BufWriter::new(file);

    // Header
    if bytes_hex > 0 {
        writeln!(w, "epoch,sim_idx,x,y,hash64,is_zero,bytes_hex")?;
    } else {
        writeln!(w, "epoch,sim_idx,x,y,hash64,is_zero")?;
    }

    for (i, path) in files.iter().enumerate() {
        if i % stride != 0 {
            continue;
        }

        let (header, soup) = load_raw_data(path)?;
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

        for abs_prog_idx in 0..total_programs {
            let start = abs_prog_idx * SINGLE_TAPE_SIZE;
            let end = start + SINGLE_TAPE_SIZE;
            if end > soup.len() {
                break;
            }
            let tape = &soup[start..end];
            let zero = is_all_zero(tape);
            if exclude_zero && zero {
                continue;
            }

            let sim_idx = abs_prog_idx / programs_per_sim;
            let local = abs_prog_idx % programs_per_sim;
            let x = local % header.grid_width;
            let y = local / header.grid_width;

            let h = hash64(tape);

            if bytes_hex > 0 {
                let hex = bytes_to_hex(&tape[..bytes_hex]);
                writeln!(w, "{},{},{},{},{:016x},{},{}", header.epoch, sim_idx, x, y, h, zero, hex)?;
            } else {
                writeln!(w, "{},{},{},{},{:016x},{}", header.epoch, sim_idx, x, y, h, zero)?;
            }
        }

        // keep IO streaming
        w.flush()?;
        println!("Exported epoch {} ({})", header.epoch, path.display());
    }

    println!("Wrote CSV: {}", out_path);
    Ok(())
}


