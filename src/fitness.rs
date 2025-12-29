//! Fitness functions for program selection
//!
//! These functions evaluate programs and return a fitness score.
//! Higher scores = more likely to survive/migrate.
//!
//! NOTE: This module is not currently used by the main simulation but is kept
//! for potential future use with island models or fitness-based selection.

#![allow(dead_code)]

use crate::bff::SINGLE_TAPE_SIZE;

/// Fitness function type
pub type FitnessFn = fn(&[u8]) -> f64;

/// No selection - all programs have equal fitness
pub fn fitness_neutral(_program: &[u8]) -> f64 {
    1.0
}

/// Favor programs with more BFF commands (active code)
pub fn fitness_command_density(program: &[u8]) -> f64 {
    let commands = program.iter().filter(|&&b| is_command(b)).count();
    commands as f64 / SINGLE_TAPE_SIZE as f64
}

/// Favor programs with balanced brackets (potentially functional loops)
pub fn fitness_balanced_brackets(program: &[u8]) -> f64 {
    let opens = program.iter().filter(|&&b| b == b'[').count();
    let closes = program.iter().filter(|&&b| b == b']').count();
    
    if opens == 0 && closes == 0 {
        return 0.1; // No loops at all
    }
    
    let balance = 1.0 - (opens as f64 - closes as f64).abs() / (opens + closes) as f64;
    let has_loops = if opens > 0 && closes > 0 { 1.0 } else { 0.5 };
    
    balance * has_loops
}

/// Favor programs with copy instructions (. and ,)
pub fn fitness_copy_instructions(program: &[u8]) -> f64 {
    let copies = program.iter().filter(|&&b| b == b'.' || b == b',').count();
    (copies as f64 / 10.0).min(1.0) // Saturate at 10 copy instructions
}

/// Favor programs with low entropy (repetitive patterns = potential replicators)
pub fn fitness_low_entropy(program: &[u8]) -> f64 {
    let mut counts = [0u32; 256];
    for &b in program {
        counts[b as usize] += 1;
    }
    
    let mut entropy = 0.0;
    let len = program.len() as f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }
    
    // Max entropy for 64 bytes ≈ 6 bits, low entropy is better
    let max_entropy = 6.0;
    1.0 - (entropy / max_entropy).min(1.0)
}

/// Favor programs with high entropy (diverse byte values)
pub fn fitness_high_entropy(program: &[u8]) -> f64 {
    1.0 - fitness_low_entropy(program)
}

/// Favor programs that start with specific patterns (e.g., self-replicator signatures)
pub fn fitness_pattern_match(program: &[u8]) -> f64 {
    // Look for common self-replicator patterns
    let patterns: &[&[u8]] = &[
        b">[",      // Forward scan loop
        b"<[",      // Backward scan loop  
        b"[->",     // Decrement and move
        b"[-<",     // Decrement and move back
        b"[.>",     // Copy forward loop
        b"[,>",     // Read forward loop
    ];
    
    let mut matches = 0;
    for pattern in patterns {
        if program.windows(pattern.len()).any(|w| w == *pattern) {
            matches += 1;
        }
    }
    
    matches as f64 / patterns.len() as f64
}

/// Combined fitness: weighted sum of multiple criteria
pub fn fitness_combined(program: &[u8]) -> f64 {
    0.3 * fitness_command_density(program)
        + 0.2 * fitness_balanced_brackets(program)
        + 0.3 * fitness_copy_instructions(program)
        + 0.2 * fitness_pattern_match(program)
}

/// Check if a byte is a BFF command
fn is_command(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'>' | b'<' | b'{' | b'}' | b'[' | b']' | b'.' | b',')
}

/// Get fitness function by name
pub fn get_fitness_fn(name: &str) -> Option<FitnessFn> {
    match name {
        "neutral" | "none" => Some(fitness_neutral),
        "command_density" | "commands" => Some(fitness_command_density),
        "balanced_brackets" | "brackets" => Some(fitness_balanced_brackets),
        "copy_instructions" | "copy" => Some(fitness_copy_instructions),
        "low_entropy" => Some(fitness_low_entropy),
        "high_entropy" => Some(fitness_high_entropy),
        "pattern_match" | "patterns" => Some(fitness_pattern_match),
        "combined" => Some(fitness_combined),
        _ => None,
    }
}

/// List available fitness functions
pub fn list_fitness_functions() -> Vec<(&'static str, &'static str)> {
    vec![
        ("neutral", "No selection pressure (default)"),
        ("command_density", "Favor programs with more BFF commands"),
        ("balanced_brackets", "Favor programs with balanced [ ] loops"),
        ("copy_instructions", "Favor programs with . and , (copy ops)"),
        ("low_entropy", "Favor repetitive patterns (potential replicators)"),
        ("high_entropy", "Favor diverse byte distributions"),
        ("pattern_match", "Favor known self-replicator patterns"),
        ("combined", "Weighted combination of multiple criteria"),
    ]
}


