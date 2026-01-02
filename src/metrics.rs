//! Metrics for tracking simulation phase transitions
//!
//! Uses Brotli compression ratio as a key metric for detecting when
//! self-replicators emerge from the primordial soup.
//!
//! Based on: "Computational Life: How Well-formed, Self-replicating Programs
//! Emerge from Simple Interaction" (Agüera y Arcas et al., 2024)
//! https://arxiv.org/pdf/2406.19108

#![allow(dead_code)] // Metrics are conditionally used based on config

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::Instant;

/// Configuration for metrics collection
#[derive(Clone, Debug)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,
    /// Interval (in epochs) between metric calculations
    pub interval: usize,
    /// Path to CSV output file (None = stdout only)
    pub output_path: Option<String>,
    /// Brotli compression quality (1-11, lower = faster)
    pub brotli_quality: u32,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: 1000,
            output_path: None,
            brotli_quality: 4, // Balance between speed and compression
        }
    }
}

/// Collected metrics for a single epoch
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub compression_ratio: f64,
    pub compressed_size: usize,
    pub original_size: usize,
    pub unique_bytes: usize,
    pub zero_byte_fraction: f64,
    pub command_fraction: f64,
    pub computation_time_ms: f64,
}

impl EpochMetrics {
    /// Format as CSV row
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{:.4},{},{},{},{:.4},{:.4},{:.2}",
            self.epoch,
            self.compression_ratio,
            self.compressed_size,
            self.original_size,
            self.unique_bytes,
            self.zero_byte_fraction,
            self.command_fraction,
            self.computation_time_ms,
        )
    }

    /// CSV header
    pub fn csv_header() -> &'static str {
        "epoch,compression_ratio,compressed_size,original_size,unique_bytes,zero_byte_fraction,command_fraction,computation_time_ms"
    }
}

/// Metrics tracker that collects and logs simulation metrics
pub struct MetricsTracker {
    config: MetricsConfig,
    csv_writer: Option<BufWriter<File>>,
    history: Vec<EpochMetrics>,
    last_ratio: f64,
    phase_transition_detected: bool,
    phase_transition_epoch: Option<usize>,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new(config: MetricsConfig) -> std::io::Result<Self> {
        let csv_writer = if let Some(ref path) = config.output_path {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", EpochMetrics::csv_header())?;
            Some(writer)
        } else {
            None
        };

        Ok(Self {
            config,
            csv_writer,
            history: Vec::new(),
            last_ratio: 1.0,
            phase_transition_detected: false,
            phase_transition_epoch: None,
        })
    }

    /// Check if we should collect metrics this epoch
    pub fn should_collect(&self, epoch: usize) -> bool {
        self.config.enabled && epoch % self.config.interval == 0
    }

    /// Collect metrics for the current soup state
    pub fn collect(&mut self, epoch: usize, soup: &[u8]) -> EpochMetrics {
        let start = Instant::now();
        
        // Calculate Brotli compression ratio
        let (compressed_size, compression_ratio) = self.calculate_compression_ratio(soup);
        
        // Count unique bytes
        let mut byte_counts = [0u32; 256];
        for &b in soup {
            byte_counts[b as usize] += 1;
        }
        let unique_bytes = byte_counts.iter().filter(|&&c| c > 0).count();
        
        // Zero byte fraction (dead/empty programs)
        let zero_count = byte_counts[0] as f64;
        let zero_byte_fraction = zero_count / soup.len() as f64;
        
        // BFF command fraction
        let command_count = soup.iter().filter(|&&b| is_bff_command(b)).count();
        let command_fraction = command_count as f64 / soup.len() as f64;
        
        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let metrics = EpochMetrics {
            epoch,
            compression_ratio,
            compressed_size,
            original_size: soup.len(),
            unique_bytes,
            zero_byte_fraction,
            command_fraction,
            computation_time_ms,
        };
        
        // Detect phase transition (significant jump in compression ratio)
        if !self.phase_transition_detected && compression_ratio > self.last_ratio * 1.5 && compression_ratio > 1.3 {
            self.phase_transition_detected = true;
            self.phase_transition_epoch = Some(epoch);
            eprintln!("\n╔══════════════════════════════════════════════════════════╗");
            eprintln!("║  🧬 PHASE TRANSITION DETECTED at epoch {}!", epoch);
            eprintln!("║  Compression ratio jumped from {:.2} to {:.2}", self.last_ratio, compression_ratio);
            eprintln!("║  Self-replicators may have emerged!", );
            eprintln!("╚══════════════════════════════════════════════════════════╝\n");
        }
        
        self.last_ratio = compression_ratio;
        
        // Log to CSV
        if let Some(ref mut writer) = self.csv_writer {
            let _ = writeln!(writer, "{}", metrics.to_csv_row());
            let _ = writer.flush();
        }
        
        self.history.push(metrics.clone());
        
        metrics
    }

    /// Calculate Brotli compression ratio
    fn calculate_compression_ratio(&self, data: &[u8]) -> (usize, f64) {
        use brotli::enc::BrotliEncoderParams;
        
        let mut compressed = Vec::new();
        let mut params = BrotliEncoderParams::default();
        params.quality = self.config.brotli_quality as i32;
        
        let result = brotli::BrotliCompress(
            &mut std::io::Cursor::new(data),
            &mut compressed,
            &params
        );
        
        match result {
            Ok(_) => {
                let ratio = data.len() as f64 / compressed.len() as f64;
                (compressed.len(), ratio)
            }
            Err(_) => (data.len(), 1.0) // Fallback if compression fails
        }
    }

    /// Get the phase transition epoch if detected
    pub fn phase_transition_epoch(&self) -> Option<usize> {
        self.phase_transition_epoch
    }

    /// Get all collected metrics history
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }

    /// Print a summary of metrics
    pub fn print_summary(&self) {
        if self.history.is_empty() {
            return;
        }

        let first = self.history.first().unwrap();
        let last = self.history.last().unwrap();
        
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│                    METRICS SUMMARY                          │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│  Epochs tracked:        {:>8} → {:>8}                  │", first.epoch, last.epoch);
        println!("│  Compression ratio:     {:>8.2} → {:>8.2}                  │", first.compression_ratio, last.compression_ratio);
        println!("│  Unique bytes:          {:>8} → {:>8}                  │", first.unique_bytes, last.unique_bytes);
        println!("│  Zero byte fraction:    {:>7.1}% → {:>7.1}%                  │", 
            first.zero_byte_fraction * 100.0, last.zero_byte_fraction * 100.0);
        println!("│  Command fraction:      {:>7.1}% → {:>7.1}%                  │",
            first.command_fraction * 100.0, last.command_fraction * 100.0);
        
        if let Some(epoch) = self.phase_transition_epoch {
            println!("├─────────────────────────────────────────────────────────────┤");
            println!("│  🧬 Phase transition detected at epoch {:>8}            │", epoch);
        }
        
        println!("└─────────────────────────────────────────────────────────────┘");
        
        if let Some(ref path) = self.config.output_path {
            println!("  Metrics saved to: {}", path);
        }
    }
}

/// Check if a byte is a BFF command
fn is_bff_command(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'>' | b'<' | b'{' | b'}' | b'[' | b']' | b'.' | b',')
}

/// Quick compression ratio calculation (for one-off checks)
pub fn quick_compression_ratio(data: &[u8]) -> f64 {
    use brotli::enc::BrotliEncoderParams;
    
    let mut compressed = Vec::new();
    let mut params = BrotliEncoderParams::default();
    params.quality = 4; // Fast but reasonable compression
    
    match brotli::BrotliCompress(
        &mut std::io::Cursor::new(data),
        &mut compressed,
        &params
    ) {
        Ok(_) => data.len() as f64 / compressed.len() as f64,
        Err(_) => 1.0
    }
}

/// Per-simulation metrics for multi-sim mode
#[derive(Clone, Debug)]
pub struct SimMetrics {
    pub sim_idx: usize,
    pub compression_ratio: f64,
    pub zero_byte_fraction: f64,
    pub command_fraction: f64,
}

/// Calculate metrics for each simulation in a multi-sim setup
pub fn calculate_per_sim_metrics(
    all_soup: &[u8],
    num_sims: usize,
    programs_per_sim: usize,
    bytes_per_program: usize,
) -> Vec<SimMetrics> {
    let bytes_per_sim = programs_per_sim * bytes_per_program;
    
    (0..num_sims)
        .map(|sim_idx| {
            let start = sim_idx * bytes_per_sim;
            let end = start + bytes_per_sim;
            let sim_soup = &all_soup[start..end];
            
            let compression_ratio = quick_compression_ratio(sim_soup);
            
            let zero_count = sim_soup.iter().filter(|&&b| b == 0).count();
            let zero_byte_fraction = zero_count as f64 / sim_soup.len() as f64;
            
            let command_count = sim_soup.iter().filter(|&&b| is_bff_command(b)).count();
            let command_fraction = command_count as f64 / sim_soup.len() as f64;
            
            SimMetrics {
                sim_idx,
                compression_ratio,
                zero_byte_fraction,
                command_fraction,
            }
        })
        .collect()
}

