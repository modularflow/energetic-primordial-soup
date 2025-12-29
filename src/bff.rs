//! BFF - Brainfuck variant with two heads
//! Matching semantics from cubff/bff.cu
//!
//! Note: The GPU backends implement BFF evaluation in shaders. The CPU
//! functions here are kept for testing, debugging, and potential CPU-only use.

#![allow(dead_code)]

pub const SINGLE_TAPE_SIZE: usize = 64;
pub const FULL_TAPE_SIZE: usize = 2 * SINGLE_TAPE_SIZE; // 128 bytes

/// Wrap a position to stay within tape bounds
#[inline]
fn wrap_pos(pos: i32) -> usize {
    (pos as usize) & (FULL_TAPE_SIZE - 1)
}

/// Determine which program "owns" a tape position
/// Returns 0 for first program (bytes 0-63), 1 for second program (bytes 64-127)
#[inline]
pub fn pos_to_program(pos: usize) -> usize {
    if pos < SINGLE_TAPE_SIZE { 0 } else { 1 }
}

/// Check if a character is a BFF command
#[inline]
pub fn is_command(c: u8) -> bool {
    matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']')
}

/// A cross-program copy event (copy between first and second half of tape)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossProgramCopy {
    /// Which half the data came FROM (0 = first 64 bytes, 1 = second 64 bytes)
    pub source_half: usize,
    /// Which half the data went TO (0 = first 64 bytes, 1 = second 64 bytes)
    pub dest_half: usize,
}

impl CrossProgramCopy {
    pub fn new(source_half: usize, dest_half: usize) -> Self {
        Self { source_half, dest_half }
    }
}

/// Result of BFF evaluation with copy tracking
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Number of non-NOP operations executed
    pub ops: usize,
    /// Cross-program copy events (copies between first and second half)
    pub cross_copies: Vec<CrossProgramCopy>,
}

/// Evaluate a BFF program on a 128-byte tape
/// Returns the number of non-NOP instructions executed
/// 
/// Tape layout (matching cubff):
/// - Bytes 0-1: Initial head positions (head0, head1)
/// - Bytes 2-127: Program/data
/// - First 64 bytes belong to "program 1", second 64 to "program 2"
pub fn evaluate(tape: &mut [u8; FULL_TAPE_SIZE], step_count: usize, debug: bool) -> usize {
    let mut nskip = 0usize;  // Count of NOPs
    
    // Instruction pointer starts at position 2 (after head storage bytes)
    let mut pos: i32 = 2;
    
    // Initial head positions from tape bytes 0 and 1
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    for i in 0..step_count {
        // Wrap head positions
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        if debug {
            print_program_internal(head0 as usize, head1 as usize, pos as usize, tape);
        }
        
        let cmd = tape[pos as usize];
        
        match cmd {
            b'<' => head0 -= 1,
            b'>' => head0 += 1,
            b'{' => head1 -= 1,
            b'}' => head1 += 1,
            b'+' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1),
            b'-' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1),
            b'.' => tape[wrap_pos(head1)] = tape[wrap_pos(head0)],
            b',' => tape[wrap_pos(head0)] = tape[wrap_pos(head1)],
            b'[' => {
                if tape[wrap_pos(head0)] == 0 {
                    // Jump forward to matching ']'
                    let mut depth = 1i32;
                    pos += 1;
                    while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth -= 1,
                            b'[' => depth += 1,
                            _ => {}
                        }
                        pos += 1;
                    }
                    pos -= 1;
                    if depth != 0 {
                        pos = FULL_TAPE_SIZE as i32; // Terminate
                    }
                }
            }
            b']' => {
                if tape[wrap_pos(head0)] != 0 {
                    // Jump backward to matching '['
                    let mut depth = 1i32;
                    pos -= 1;
                    while pos >= 0 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth += 1,
                            b'[' => depth -= 1,
                            _ => {}
                        }
                        pos -= 1;
                    }
                    pos += 1;
                    if depth != 0 {
                        pos = -1; // Terminate
                    }
                }
            }
            _ => nskip += 1,
        }
        
        // Check termination conditions
        if pos < 0 {
            return i + 1 - nskip;
        }
        
        pos += 1;
        
        if pos >= FULL_TAPE_SIZE as i32 {
            return i + 1 - nskip;
        }
    }
    
    step_count - nskip
}

/// Evaluate a BFF program and track cross-program copy operations
/// 
/// This variant tracks when `.` or `,` copies data between the two program halves:
/// - First half (bytes 0-63): program 1
/// - Second half (bytes 64-127): program 2
/// 
/// Returns both the operation count and a list of cross-program copies.
pub fn evaluate_with_copy_tracking(
    tape: &mut [u8; FULL_TAPE_SIZE],
    step_count: usize,
) -> EvalResult {
    let mut nskip = 0usize;
    let mut cross_copies = Vec::new();
    
    let mut pos: i32 = 2;
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    for i in 0..step_count {
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        let cmd = tape[pos as usize];
        
        match cmd {
            b'<' => head0 -= 1,
            b'>' => head0 += 1,
            b'{' => head1 -= 1,
            b'}' => head1 += 1,
            b'+' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1),
            b'-' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1),
            b'.' => {
                // Copy from head0 to head1
                let src_pos = wrap_pos(head0);
                let dst_pos = wrap_pos(head1);
                tape[dst_pos] = tape[src_pos];
                
                // Track if this is a cross-program copy
                let src_half = pos_to_program(src_pos);
                let dst_half = pos_to_program(dst_pos);
                if src_half != dst_half {
                    cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                }
            }
            b',' => {
                // Copy from head1 to head0
                let src_pos = wrap_pos(head1);
                let dst_pos = wrap_pos(head0);
                tape[dst_pos] = tape[src_pos];
                
                // Track if this is a cross-program copy
                let src_half = pos_to_program(src_pos);
                let dst_half = pos_to_program(dst_pos);
                if src_half != dst_half {
                    cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                }
            }
            b'[' => {
                if tape[wrap_pos(head0)] == 0 {
                    let mut depth = 1i32;
                    pos += 1;
                    while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth -= 1,
                            b'[' => depth += 1,
                            _ => {}
                        }
                        pos += 1;
                    }
                    pos -= 1;
                    if depth != 0 {
                        pos = FULL_TAPE_SIZE as i32;
                    }
                }
            }
            b']' => {
                if tape[wrap_pos(head0)] != 0 {
                    let mut depth = 1i32;
                    pos -= 1;
                    while pos >= 0 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth += 1,
                            b'[' => depth -= 1,
                            _ => {}
                        }
                        pos -= 1;
                    }
                    pos += 1;
                    if depth != 0 {
                        pos = -1;
                    }
                }
            }
            _ => nskip += 1,
        }
        
        if pos < 0 {
            return EvalResult {
                ops: i + 1 - nskip,
                cross_copies,
            };
        }
        
        pos += 1;
        
        if pos >= FULL_TAPE_SIZE as i32 {
            return EvalResult {
                ops: i + 1 - nskip,
                cross_copies,
            };
        }
    }
    
    EvalResult {
        ops: step_count - nskip,
        cross_copies,
    }
}

/// Pretty-print the current state (for debugging)
fn print_program_internal(head0: usize, head1: usize, pc: usize, tape: &[u8; FULL_TAPE_SIZE]) {
    for (i, &byte) in tape.iter().enumerate() {
        let c = if byte.is_ascii_graphic() || byte == b' ' {
            byte as char
        } else if byte == 0 {
            '␀'
        } else {
            ' '
        };
        
        // Color coding via ANSI escape codes
        if i == head0 {
            print!("\x1b[44;1m"); // Blue background
        }
        if i == head1 {
            print!("\x1b[41;1m"); // Red background  
        }
        if i == pc {
            print!("\x1b[42;1m"); // Green background
        }
        if is_command(byte) {
            print!("\x1b[37;1m"); // Bright white
        }
        
        print!("{}", c);
        
        if is_command(byte) || i == head0 || i == head1 || i == pc {
            print!("\x1b[;m"); // Reset
        }
    }
    println!();
}

/// Create a tape from a program string
pub fn parse_program(program: &str) -> [u8; FULL_TAPE_SIZE] {
    let mut tape = [0u8; FULL_TAPE_SIZE];
    for (i, byte) in program.bytes().take(FULL_TAPE_SIZE).enumerate() {
        tape[i] = byte;
    }
    tape
}

/// Print program as readable string
pub fn tape_to_string(tape: &[u8; FULL_TAPE_SIZE]) -> String {
    tape.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else if b == 0 {
                '␀'
            } else {
                ' '
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_increment() {
        // Program: increment cell at head0
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';
        tape[3] = b'+';
        tape[4] = b'+';
        
        let ops = evaluate(&mut tape, 100, false);
        
        // head0 starts at position 0 (tape[0] = 0)
        assert_eq!(tape[0], 3); // Three increments
        assert_eq!(ops, 3);
    }
    
    #[test]
    fn test_copy_operation() {
        // Set up tape with value at head0, copy to head1
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 starts at position 10
        tape[1] = 20; // head1 starts at position 20
        tape[10] = 42; // Value to copy
        tape[2] = b'.'; // Copy from head0 to head1
        
        let ops = evaluate(&mut tape, 100, false);
        
        assert_eq!(tape[20], 42);
        assert_eq!(ops, 1);
    }
    
    #[test]
    fn test_simple_loop() {
        // Program: [->+<] - move value from cell 0 to cell 1
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 2; // head0 at position 2 (will be overwritten by program)
        // Actually, let's start head0 at a data area
        tape[0] = 64; // head0 at position 64
        tape[1] = 65; // head1 at position 65
        tape[64] = 3; // Value to move
        tape[2] = b'[';
        tape[3] = b'-';
        tape[4] = b'>';
        tape[5] = b'+';
        tape[6] = b'<';
        tape[7] = b']';
        
        evaluate(&mut tape, 1000, false);
        
        assert_eq!(tape[64], 0); // Source cleared
        assert_eq!(tape[65], 3); // Destination has value
    }
    
    #[test]
    fn test_pos_to_program() {
        // First half (0-63) belongs to program 0
        assert_eq!(pos_to_program(0), 0);
        assert_eq!(pos_to_program(32), 0);
        assert_eq!(pos_to_program(63), 0);
        
        // Second half (64-127) belongs to program 1
        assert_eq!(pos_to_program(64), 1);
        assert_eq!(pos_to_program(96), 1);
        assert_eq!(pos_to_program(127), 1);
    }
    
    #[test]
    fn test_cross_copy_tracking_same_half() {
        // Copy within the same half - should NOT be tracked as cross-program
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 20; // head1 at position 20 (first half)
        tape[10] = 42;
        tape[2] = b'.'; // Copy from 10 to 20 (both in first half)
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[20], 42);
        assert_eq!(result.ops, 1);
        assert!(result.cross_copies.is_empty(), "No cross-program copies expected");
    }
    
    #[test]
    fn test_cross_copy_tracking_first_to_second() {
        // Copy from first half to second half - should be tracked
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 80; // head1 at position 80 (second half)
        tape[10] = 42;
        tape[2] = b'.'; // Copy from 10 to 80
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[80], 42);
        assert_eq!(result.ops, 1);
        assert_eq!(result.cross_copies.len(), 1);
        assert_eq!(result.cross_copies[0].source_half, 0);
        assert_eq!(result.cross_copies[0].dest_half, 1);
    }
    
    #[test]
    fn test_cross_copy_tracking_second_to_first() {
        // Copy from second half to first half using ','
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 80; // head1 at position 80 (second half)
        tape[80] = 99;
        tape[2] = b','; // Copy from 80 to 10
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[10], 99);
        assert_eq!(result.ops, 1);
        assert_eq!(result.cross_copies.len(), 1);
        assert_eq!(result.cross_copies[0].source_half, 1);
        assert_eq!(result.cross_copies[0].dest_half, 0);
    }
    
    #[test]
    fn test_cross_copy_tracking_multiple() {
        // Multiple cross-program copies in a loop
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 starts at 10 (first half)
        tape[1] = 80; // head1 starts at 80 (second half)
        tape[10] = 3; // Loop 3 times
        // Program: [-.}] - copy and decrement, moving head1 forward
        tape[2] = b'[';
        tape[3] = b'-';
        tape[4] = b'.';
        tape[5] = b'}';
        tape[6] = b']';
        
        let result = evaluate_with_copy_tracking(&mut tape, 1000);
        
        // Should have 3 cross-program copies (one per loop iteration)
        assert_eq!(result.cross_copies.len(), 3);
        for copy in &result.cross_copies {
            assert_eq!(copy.source_half, 0);
            assert_eq!(copy.dest_half, 1);
        }
    }
}
