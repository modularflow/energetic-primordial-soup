//! Checkpoint system for saving and restoring simulation state
//!
//! Checkpoints capture the complete simulation state including:
//! - Soup data (all program tapes)
//! - Energy states (reserve, timer, dead status per program)
//! - Current epoch
//! - Configuration for validation
//!
//! Note: Some utility functions are kept for future use even if not currently called.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Checkpoint header with metadata and config validation info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Format version for forward compatibility
    pub version: u32,
    /// Current epoch when checkpoint was saved
    pub epoch: usize,
    /// Grid dimensions
    pub grid_width: usize,
    pub grid_height: usize,
    /// Number of parallel simulations
    pub num_sims: usize,
    /// Layout of parallel sims [cols, rows]
    pub parallel_layout: [usize; 2],
    /// Whether border interaction was enabled
    pub border_interaction: bool,
    /// Timestamp when checkpoint was saved
    pub timestamp: u64,
    /// Original seed (for info)
    pub seed: u64,
    /// Number of programs per simulation
    pub programs_per_sim: usize,
}

/// Complete checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub header: CheckpointHeader,
    /// Soup data for all simulations (concatenated)
    /// Size: programs_per_sim * 64 * num_sims bytes
    pub soup: Vec<u8>,
    /// Energy states for all simulations (concatenated)
    /// Size: programs_per_sim * num_sims u32s (packed: reserve|timer|dead|unused)
    pub energy_states: Vec<u32>,
}

impl Checkpoint {
    /// Create a new checkpoint from simulation state
    pub fn new(
        epoch: usize,
        grid_width: usize,
        grid_height: usize,
        num_sims: usize,
        parallel_layout: [usize; 2],
        border_interaction: bool,
        seed: u64,
        soup: Vec<u8>,
        energy_states: Vec<u32>,
    ) -> Self {
        let programs_per_sim = grid_width * grid_height;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            header: CheckpointHeader {
                version: 1,
                epoch,
                grid_width,
                grid_height,
                num_sims,
                parallel_layout,
                border_interaction,
                timestamp,
                seed,
                programs_per_sim,
            },
            soup,
            energy_states,
        }
    }

    /// Save checkpoint to file (binary format for efficiency)
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(b"BFFCK")?;

        // Serialize header as YAML (for easy inspection)
        let header_yaml = serde_yaml::to_string(&self.header)?;
        let header_len = header_yaml.len() as u32;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(header_yaml.as_bytes())?;

        // Write soup data (raw bytes)
        let soup_len = self.soup.len() as u64;
        writer.write_all(&soup_len.to_le_bytes())?;
        writer.write_all(&self.soup)?;

        // Write energy states (as raw u32 bytes)
        let energy_len = self.energy_states.len() as u64;
        writer.write_all(&energy_len.to_le_bytes())?;
        let energy_bytes: Vec<u8> = self
            .energy_states
            .iter()
            .flat_map(|&e| e.to_le_bytes())
            .collect();
        writer.write_all(&energy_bytes)?;

        writer.flush()?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Check magic bytes
        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic)?;
        if &magic != b"BFFCK" {
            return Err("Invalid checkpoint file: bad magic bytes".into());
        }

        // Read header
        let mut header_len_bytes = [0u8; 4];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;

        let mut header_yaml = vec![0u8; header_len];
        reader.read_exact(&mut header_yaml)?;
        let header: CheckpointHeader = serde_yaml::from_slice(&header_yaml)?;

        // Read soup
        let mut soup_len_bytes = [0u8; 8];
        reader.read_exact(&mut soup_len_bytes)?;
        let soup_len = u64::from_le_bytes(soup_len_bytes) as usize;

        let mut soup = vec![0u8; soup_len];
        reader.read_exact(&mut soup)?;

        // Read energy states
        let mut energy_len_bytes = [0u8; 8];
        reader.read_exact(&mut energy_len_bytes)?;
        let energy_len = u64::from_le_bytes(energy_len_bytes) as usize;

        let mut energy_bytes = vec![0u8; energy_len * 4];
        reader.read_exact(&mut energy_bytes)?;

        let energy_states: Vec<u32> = energy_bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self {
            header,
            soup,
            energy_states,
        })
    }

    /// Validate checkpoint against current config
    pub fn validate(
        &self,
        grid_width: usize,
        grid_height: usize,
        num_sims: usize,
        parallel_layout: [usize; 2],
    ) -> Result<(), String> {
        if self.header.grid_width != grid_width {
            return Err(format!(
                "Grid width mismatch: checkpoint={}, config={}",
                self.header.grid_width, grid_width
            ));
        }
        if self.header.grid_height != grid_height {
            return Err(format!(
                "Grid height mismatch: checkpoint={}, config={}",
                self.header.grid_height, grid_height
            ));
        }
        if self.header.num_sims != num_sims {
            return Err(format!(
                "Number of sims mismatch: checkpoint={}, config={}",
                self.header.num_sims, num_sims
            ));
        }
        if self.header.parallel_layout != parallel_layout {
            return Err(format!(
                "Parallel layout mismatch: checkpoint={:?}, config={:?}",
                self.header.parallel_layout, parallel_layout
            ));
        }

        // Validate data sizes
        let expected_soup_size = self.header.programs_per_sim * 64 * num_sims;
        if self.soup.len() != expected_soup_size {
            return Err(format!(
                "Soup size mismatch: got={}, expected={}",
                self.soup.len(),
                expected_soup_size
            ));
        }

        let expected_energy_size = self.header.programs_per_sim * num_sims;
        if self.energy_states.len() != expected_energy_size {
            return Err(format!(
                "Energy state size mismatch: got={}, expected={}",
                self.energy_states.len(),
                expected_energy_size
            ));
        }

        Ok(())
    }

    /// Get soup data for a specific simulation
    pub fn get_sim_soup(&self, sim_idx: usize) -> &[u8] {
        let programs = self.header.programs_per_sim;
        let start = sim_idx * programs * 64;
        let end = start + programs * 64;
        &self.soup[start..end]
    }

    /// Get energy states for a specific simulation
    pub fn get_sim_energy(&self, sim_idx: usize) -> &[u32] {
        let programs = self.header.programs_per_sim;
        let start = sim_idx * programs;
        let end = start + programs;
        &self.energy_states[start..end]
    }
}

/// Generate checkpoint filename with epoch
pub fn checkpoint_filename(base_dir: &str, epoch: usize, num_sims: usize) -> String {
    if num_sims > 1 {
        format!("{}/checkpoint_epoch_{}_sims_{}.bff", base_dir, epoch, num_sims)
    } else {
        format!("{}/checkpoint_epoch_{}.bff", base_dir, epoch)
    }
}

/// Find the latest checkpoint in a directory
pub fn find_latest_checkpoint(base_dir: &str) -> Option<String> {
    let path = Path::new(base_dir);
    if !path.exists() {
        return None;
    }

    let mut latest: Option<(usize, String)> = None;

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("checkpoint_epoch_") && name.ends_with(".bff") {
                    // Extract epoch number
                    if let Some(epoch_str) = name
                        .strip_prefix("checkpoint_epoch_")
                        .and_then(|s| s.split('_').next())
                        .and_then(|s| s.strip_suffix(".bff").or(Some(s)))
                    {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            if latest.is_none() || epoch > latest.as_ref().unwrap().0 {
                                latest = Some((epoch, entry.path().to_string_lossy().to_string()));
                            }
                        }
                    }
                }
            }
        }
    }

    latest.map(|(_, path)| path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_round_trip() {
        let soup = vec![42u8; 64 * 100];
        let energy = vec![0u32; 100];

        let checkpoint = Checkpoint::new(
            1000,
            10,
            10,
            1,
            [1, 1],
            false,
            42,
            soup.clone(),
            energy.clone(),
        );

        let path = "/tmp/test_checkpoint.bff";
        checkpoint.save(path).unwrap();

        let loaded = Checkpoint::load(path).unwrap();
        assert_eq!(loaded.header.epoch, 1000);
        assert_eq!(loaded.soup, soup);
        assert_eq!(loaded.energy_states, energy);

        // Cleanup
        let _ = fs::remove_file(path);
    }
}

