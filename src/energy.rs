//! Energy Sources System
//!
//! Provides localized "energy sources" that control which programs can mutate.
//! Programs within range of an energy source are energized and can mutate.
//! Programs outside must rely on reserve energy or risk death.
//!
//! Key mechanics:
//! - Energy sources are fixed points on the grid with a radius/size
//! - Programs leaving energy zones get reserve energy (default: 5 epochs)
//! - Programs without interaction for too long die (default: 10 epochs)
//! - Copy operations transfer energy to recipients
//! - Sources can have lifetimes and spawn dynamically
//! - Dead tapes in energy zones can spontaneously generate new programs
//!
//! Note: GPU backends reimplement energy logic in shaders. CPU functions
//! here are kept for the CPU backend and testing.

#![allow(dead_code)]

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Shape of an energy source
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyShape {
    /// Circular zone (radius in all directions)
    Circle,
    /// Horizontal strip (width = 2*radius, height = radius/2)
    StripHorizontal,
    /// Vertical strip (width = radius/2, height = 2*radius)
    StripVertical,
    /// Half circle (top half of a circle)
    HalfCircleTop,
    /// Half circle (bottom half)
    HalfCircleBottom,
    /// Half circle (left half)
    HalfCircleLeft,
    /// Half circle (right half)
    HalfCircleRight,
    /// Ellipse (width = 2*radius, height = radius)
    EllipseHorizontal,
    /// Ellipse (width = radius, height = 2*radius)
    EllipseVertical,
}

impl EnergyShape {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "circle" => EnergyShape::Circle,
            "strip_h" | "strip_horizontal" | "horizontal_strip" => EnergyShape::StripHorizontal,
            "strip_v" | "strip_vertical" | "vertical_strip" => EnergyShape::StripVertical,
            "half_circle" | "half_circle_top" => EnergyShape::HalfCircleTop,
            "half_circle_bottom" => EnergyShape::HalfCircleBottom,
            "half_circle_left" => EnergyShape::HalfCircleLeft,
            "half_circle_right" => EnergyShape::HalfCircleRight,
            "ellipse" | "ellipse_h" | "ellipse_horizontal" => EnergyShape::EllipseHorizontal,
            "ellipse_v" | "ellipse_vertical" => EnergyShape::EllipseVertical,
            "random" => EnergyShape::Circle, // Random is handled at source creation
            _ => EnergyShape::Circle,
        }
    }
    
    /// Get a random shape
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..9) {
            0 => EnergyShape::Circle,
            1 => EnergyShape::StripHorizontal,
            2 => EnergyShape::StripVertical,
            3 => EnergyShape::HalfCircleTop,
            4 => EnergyShape::HalfCircleBottom,
            5 => EnergyShape::HalfCircleLeft,
            6 => EnergyShape::HalfCircleRight,
            7 => EnergyShape::EllipseHorizontal,
            _ => EnergyShape::EllipseVertical,
        }
    }
    
    /// Convert to GPU representation (0-8)
    pub fn to_gpu_id(&self) -> u32 {
        match self {
            EnergyShape::Circle => 0,
            EnergyShape::StripHorizontal => 1,
            EnergyShape::StripVertical => 2,
            EnergyShape::HalfCircleTop => 3,
            EnergyShape::HalfCircleBottom => 4,
            EnergyShape::HalfCircleLeft => 5,
            EnergyShape::HalfCircleRight => 6,
            EnergyShape::EllipseHorizontal => 7,
            EnergyShape::EllipseVertical => 8,
        }
    }
}

/// An energy source on the grid
#[derive(Clone, Debug, PartialEq)]
pub struct EnergySource {
    pub x: usize,
    pub y: usize,
    pub radius: usize,
    pub shape: EnergyShape,
    /// Age in epochs (for lifetime tracking)
    pub age: usize,
}

impl EnergySource {
    /// Create a new circular energy source
    pub fn new(x: usize, y: usize, radius: usize) -> Self {
        Self { x, y, radius, shape: EnergyShape::Circle, age: 0 }
    }
    
    /// Create a new energy source with a specific shape
    pub fn with_shape(x: usize, y: usize, radius: usize, shape: EnergyShape) -> Self {
        Self { x, y, radius, shape, age: 0 }
    }

    /// Check if a position is within this source's range based on shape
    pub fn contains(&self, x: usize, y: usize) -> bool {
        let dx = x as f64 - self.x as f64;
        let dy = y as f64 - self.y as f64;
        let r = self.radius as f64;
        
        match self.shape {
            EnergyShape::Circle => {
                dx * dx + dy * dy <= r * r
            }
            EnergyShape::StripHorizontal => {
                // Width = 2*radius, Height = radius/2
                dx.abs() <= r && dy.abs() <= r / 4.0
            }
            EnergyShape::StripVertical => {
                // Width = radius/2, Height = 2*radius
                dx.abs() <= r / 4.0 && dy.abs() <= r
            }
            EnergyShape::HalfCircleTop => {
                dy <= 0.0 && dx * dx + dy * dy <= r * r
            }
            EnergyShape::HalfCircleBottom => {
                dy >= 0.0 && dx * dx + dy * dy <= r * r
            }
            EnergyShape::HalfCircleLeft => {
                dx <= 0.0 && dx * dx + dy * dy <= r * r
            }
            EnergyShape::HalfCircleRight => {
                dx >= 0.0 && dx * dx + dy * dy <= r * r
            }
            EnergyShape::EllipseHorizontal => {
                // Width = 2*radius, Height = radius
                (dx / r).powi(2) + (dy / (r / 2.0)).powi(2) <= 1.0
            }
            EnergyShape::EllipseVertical => {
                // Width = radius, Height = 2*radius
                (dx / (r / 2.0)).powi(2) + (dy / r).powi(2) <= 1.0
            }
        }
    }

    /// Get squared distance (for comparisons without sqrt)
    pub fn distance_squared(&self, x: usize, y: usize) -> f64 {
        let dx = x as f64 - self.x as f64;
        let dy = y as f64 - self.y as f64;
        dx * dx + dy * dy
    }
    
    /// Increment age, returns true if source should expire
    pub fn tick(&mut self, lifetime: usize) -> bool {
        self.age += 1;
        lifetime > 0 && self.age >= lifetime
    }
}

/// Dynamic energy source configuration
#[derive(Clone, Debug)]
pub struct DynamicEnergyConfig {
    /// Use random placement instead of fixed positions
    pub random_placement: bool,
    /// Maximum number of sources that can exist
    pub max_sources: usize,
    /// Epochs until a source expires (0 = infinite)
    pub source_lifetime: usize,
    /// Spawn a new source every N epochs (0 = disabled)
    pub spawn_rate: usize,
}

impl Default for DynamicEnergyConfig {
    fn default() -> Self {
        Self {
            random_placement: false,
            max_sources: 8,
            source_lifetime: 0,
            spawn_rate: 0,
        }
    }
}

/// Energy system configuration
#[derive(Clone, Debug)]
pub struct EnergyConfig {
    /// Whether the energy system is enabled
    pub enabled: bool,
    /// List of energy sources
    pub sources: Vec<EnergySource>,
    /// Reserve epochs granted when leaving an energy zone (default: 5)
    pub reserve_duration: u32,
    /// Epochs without interaction before death (default: 10, 0 = infinite/never)
    pub interaction_death: u32,
    /// Default radius for new sources
    pub default_radius: usize,
    /// Default shape for sources ("circle", "random", etc)
    pub default_shape: String,
    /// Spontaneous generation rate: 1 in N chance per dead tape in energy zone (0 = disabled)
    pub spontaneous_rate: u32,
    /// Dynamic source options
    pub dynamic: DynamicEnergyConfig,
    /// Thickness of "dead zone" border (for mega-sim simulations)
    pub border_thickness: usize,
    /// Grid dimensions (for random placement)
    pub grid_width: usize,
    pub grid_height: usize,
    /// RNG for random placement
    rng: StdRng,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sources: Vec::new(),
            reserve_duration: 5,
            interaction_death: 10,
            default_radius: 64,
            default_shape: "circle".to_string(),
            spontaneous_rate: 0,
            dynamic: DynamicEnergyConfig::default(),
            border_thickness: 0,
            grid_width: 512,
            grid_height: 256,
            rng: StdRng::seed_from_u64(42),
        }
    }
}

impl EnergyConfig {
    /// Create a disabled energy config (original simulation behavior)
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create a configuration with N sources (1-8 supported)
    /// 
    /// Source placement:
    /// - 1: center
    /// - 2: left-center, right-center
    /// - 3: center + top-left, bottom-right
    /// - 4: 4 corners
    /// - 5: 4 corners + center
    /// - 6: 4 corners + top-center, bottom-center
    /// - 7: 4 corners + center + left-center, right-center
    /// - 8: 4 corners + 4 edge centers
    pub fn with_sources(
        width: usize,
        height: usize,
        radius: usize,
        count: usize,
        reserve_duration: u32,
        interaction_death: u32,
    ) -> Self {
        Self::with_sources_and_shape(width, height, radius, count, reserve_duration, interaction_death, EnergyShape::Circle)
    }
    
    /// Create a configuration with N sources and a specific shape
    pub fn with_sources_and_shape(
        width: usize,
        height: usize,
        radius: usize,
        count: usize,
        reserve_duration: u32,
        interaction_death: u32,
        shape: EnergyShape,
    ) -> Self {
        let r = radius;
        let cx = width / 2;
        let cy = height / 2;
        let left = r;
        let right = width.saturating_sub(r + 1);
        let top = r;
        let bottom = height.saturating_sub(r + 1);

        let mut sources = Vec::new();
        
        match count {
            0 => {}
            1 => {
                // Center only
                sources.push(EnergySource::with_shape(cx, cy, r, shape));
            }
            2 => {
                // Left and right centers
                sources.push(EnergySource::with_shape(left, cy, r, shape));
                sources.push(EnergySource::with_shape(right, cy, r, shape));
            }
            3 => {
                // Center + two diagonal corners
                sources.push(EnergySource::with_shape(cx, cy, r, shape));
                sources.push(EnergySource::with_shape(left, top, r, shape));
                sources.push(EnergySource::with_shape(right, bottom, r, shape));
            }
            4 => {
                // 4 corners (default)
                sources.push(EnergySource::with_shape(left, top, r, shape));      // Top-left
                sources.push(EnergySource::with_shape(right, top, r, shape));     // Top-right
                sources.push(EnergySource::with_shape(left, bottom, r, shape));   // Bottom-left
                sources.push(EnergySource::with_shape(right, bottom, r, shape));  // Bottom-right
            }
            5 => {
                // 4 corners + center
                sources.push(EnergySource::with_shape(left, top, r, shape));
                sources.push(EnergySource::with_shape(right, top, r, shape));
                sources.push(EnergySource::with_shape(left, bottom, r, shape));
                sources.push(EnergySource::with_shape(right, bottom, r, shape));
                sources.push(EnergySource::with_shape(cx, cy, r, shape));
            }
            6 => {
                // 4 corners + top/bottom centers
                sources.push(EnergySource::with_shape(left, top, r, shape));
                sources.push(EnergySource::with_shape(right, top, r, shape));
                sources.push(EnergySource::with_shape(left, bottom, r, shape));
                sources.push(EnergySource::with_shape(right, bottom, r, shape));
                sources.push(EnergySource::with_shape(cx, top, r, shape));
                sources.push(EnergySource::with_shape(cx, bottom, r, shape));
            }
            7 => {
                // 4 corners + center + left/right centers  
                sources.push(EnergySource::with_shape(left, top, r, shape));
                sources.push(EnergySource::with_shape(right, top, r, shape));
                sources.push(EnergySource::with_shape(left, bottom, r, shape));
                sources.push(EnergySource::with_shape(right, bottom, r, shape));
                sources.push(EnergySource::with_shape(cx, cy, r, shape));
                sources.push(EnergySource::with_shape(left, cy, r, shape));
                sources.push(EnergySource::with_shape(right, cy, r, shape));
            }
            _ => {
                // 8: 4 corners + 4 edge centers
                sources.push(EnergySource::with_shape(left, top, r, shape));
                sources.push(EnergySource::with_shape(right, top, r, shape));
                sources.push(EnergySource::with_shape(left, bottom, r, shape));
                sources.push(EnergySource::with_shape(right, bottom, r, shape));
                sources.push(EnergySource::with_shape(cx, top, r, shape));
                sources.push(EnergySource::with_shape(cx, bottom, r, shape));
                sources.push(EnergySource::with_shape(left, cy, r, shape));
                sources.push(EnergySource::with_shape(right, cy, r, shape));
            }
        }

        Self {
            enabled: count > 0,
            sources,
            reserve_duration,
            interaction_death,
            default_radius: radius,
            default_shape: "circle".to_string(),
            spontaneous_rate: 0,
            dynamic: DynamicEnergyConfig::default(),
            border_thickness: 0,
            grid_width: width,
            grid_height: height,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Create a full configuration with all options
    pub fn full(
        width: usize,
        height: usize,
        radius: usize,
        count: usize,
        reserve_duration: u32,
        interaction_death: u32,
        random_placement: bool,
        max_sources: usize,
        source_lifetime: usize,
        spawn_rate: usize,
        seed: u64,
        border_thickness: usize,
    ) -> Self {
        Self::full_with_options(
            width, height, radius, count, reserve_duration, interaction_death,
            random_placement, max_sources, source_lifetime, spawn_rate, seed,
            0, "circle".to_string(), border_thickness,
        )
    }
    
    /// Create a full configuration with all options including spontaneous generation and shape
    pub fn full_with_options(
        width: usize,
        height: usize,
        radius: usize,
        count: usize,
        reserve_duration: u32,
        interaction_death: u32,
        random_placement: bool,
        max_sources: usize,
        source_lifetime: usize,
        spawn_rate: usize,
        seed: u64,
        spontaneous_rate: u32,
        shape: String,
        border_thickness: usize,
    ) -> Self {
        let use_random_shape = shape.to_lowercase() == "random";
        let mut rng = StdRng::seed_from_u64(seed);
        
        let mut config = if random_placement {
            // Start with empty sources, will be randomly placed
            Self {
                enabled: true,
                sources: Vec::new(),
                reserve_duration,
                interaction_death,
                default_radius: radius,
                default_shape: shape.clone(),
                spontaneous_rate,
                dynamic: DynamicEnergyConfig {
                    random_placement: true,
                    max_sources,
                    source_lifetime,
                    spawn_rate,
                },
                border_thickness,
                grid_width: width,
                grid_height: height,
                rng,
            }
        } else {
            let source_shape = if use_random_shape {
                EnergyShape::random(&mut rng)
            } else {
                EnergyShape::from_str(&shape)
            };
            let mut c = Self::with_sources_and_shape(width, height, radius, count, reserve_duration, interaction_death, source_shape);
            c.default_shape = shape.clone();
            c.spontaneous_rate = spontaneous_rate;
            c.dynamic = DynamicEnergyConfig {
                random_placement: false,
                max_sources,
                source_lifetime,
                spawn_rate,
            };
            c.border_thickness = border_thickness;
            c.rng = rng;
            c
        };
        
        // Ensure fixed sources don't start inside the border dead zone.
        // (Border cells are never energized anyway, but keeping sources out avoids confusing layouts.)
        if border_thickness > 0 && !config.sources.is_empty() {
            let max_x = width.saturating_sub(border_thickness).saturating_sub(1);
            let max_y = height.saturating_sub(border_thickness).saturating_sub(1);
            for src in &mut config.sources {
                src.x = src.x.clamp(border_thickness, max_x);
                src.y = src.y.clamp(border_thickness, max_y);
            }
        }

        // If random placement, spawn initial sources
        if random_placement {
            for _ in 0..count.min(max_sources) {
                config.spawn_random_source();
            }
        }
        
        config
    }

    /// Create a 4-corner configuration with sources offset inward by radius
    pub fn corners(width: usize, height: usize, radius: usize) -> Self {
        Self::with_sources(width, height, radius, 4, 5, 10)
    }

    /// Create a custom configuration
    pub fn custom(sources: Vec<EnergySource>, reserve_duration: u32, interaction_death: u32) -> Self {
        Self {
            enabled: true,
            sources,
            reserve_duration,
            interaction_death,
            default_radius: 64,
            default_shape: "circle".to_string(),
            spontaneous_rate: 0,
            dynamic: DynamicEnergyConfig::default(),
            border_thickness: 0,
            grid_width: 512,
            grid_height: 256,
            rng: StdRng::seed_from_u64(42),
        }
    }
    
    /// Spawn a new source at a random position with potentially random shape
    pub fn spawn_random_source(&mut self) -> bool {
        if self.sources.len() >= self.dynamic.max_sources {
            return false;
        }
        
        let r = self.default_radius;
        let bt = self.border_thickness;
        
        // Validate that grid is large enough for the radius
        // Need at least 2*radius + 1 for valid placement range
        if self.grid_width <= 2 * r || self.grid_height <= 2 * r {
            // Grid too small for this radius - place at center instead
            let x = self.grid_width / 2;
            let y = self.grid_height / 2;
            
            let shape = if self.default_shape.to_lowercase() == "random" {
                EnergyShape::random(&mut self.rng)
            } else {
                EnergyShape::from_str(&self.default_shape)
            };
            
            self.sources.push(EnergySource::with_shape(x, y, r, shape));
            return true;
        }
        
        // Keep sources away from edges by radius AND out of the border dead zone.
        // Border cells are never energized, but we also prevent source centers from spawning there.
        let min_x = r + bt;
        let min_y = r + bt;
        let max_x = self.grid_width.saturating_sub(r + bt);
        let max_y = self.grid_height.saturating_sub(r + bt);

        if min_x >= max_x || min_y >= max_y {
            // No valid placement range given radius + border thickness; place at center.
            let x = self.grid_width / 2;
            let y = self.grid_height / 2;
            let shape = if self.default_shape.to_lowercase() == "random" {
                EnergyShape::random(&mut self.rng)
            } else {
                EnergyShape::from_str(&self.default_shape)
            };
            self.sources.push(EnergySource::with_shape(x, y, r, shape));
            return true;
        }

        let x = self.rng.random_range(min_x..max_x);
        let y = self.rng.random_range(min_y..max_y);
        
        // Determine shape - if "random", pick a random shape
        let shape = if self.default_shape.to_lowercase() == "random" {
            EnergyShape::random(&mut self.rng)
        } else {
            EnergyShape::from_str(&self.default_shape)
        };
        
        self.sources.push(EnergySource::with_shape(x, y, r, shape));
        true
    }
    
    /// Update sources for a new epoch (handle lifetimes and spawning)
    /// Returns true if sources changed (GPU needs buffer update)
    pub fn update_sources(&mut self, current_epoch: usize) -> bool {
        if !self.enabled {
            return false;
        }
        
        let mut changed = false;
        
        // Update ages and remove expired sources
        if self.dynamic.source_lifetime > 0 {
            let lifetime = self.dynamic.source_lifetime;
            let before = self.sources.len();
            self.sources.retain_mut(|src| !src.tick(lifetime));
            if self.sources.len() != before {
                changed = true;
            }
        }
        
        // Spawn new sources based on spawn rate
        if self.dynamic.spawn_rate > 0 && current_epoch > 0 && current_epoch % self.dynamic.spawn_rate == 0 {
            if self.spawn_random_source() {
                changed = true;
            }
        }
        
        changed
    }
    
    /// Get current number of active sources
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }
    
    /// Check if dynamic mode is active (sources can change)
    pub fn is_dynamic(&self) -> bool {
        self.dynamic.source_lifetime > 0 || self.dynamic.spawn_rate > 0
    }

    /// Freeze dynamic behavior so sources stay static for the full simulation.
    pub fn freeze_dynamic(&mut self) {
        self.dynamic.source_lifetime = 0;
        self.dynamic.spawn_rate = 0;
    }

    /// Check if a position is within any energy zone
    pub fn in_energy_zone(&self, x: usize, y: usize) -> bool {
        if !self.enabled {
            return true; // If disabled, everywhere is "energized"
        }
        // Check if we are in the "dead zone" border
        if self.border_thickness > 0 {
            if x < self.border_thickness || x >= self.grid_width - self.border_thickness ||
               y < self.border_thickness || y >= self.grid_height - self.border_thickness {
                return false; // In dead zone - no energy
            }
        }
        self.sources.iter().any(|src| src.contains(x, y))
    }

    /// Get the grid position from a program index
    pub fn idx_to_pos(&self, idx: usize, grid_width: usize) -> (usize, usize) {
        let x = idx % grid_width;
        let y = idx / grid_width;
        (x, y)
    }

    /// Check if a program index is within any energy zone
    pub fn idx_in_energy_zone(&self, idx: usize, grid_width: usize) -> bool {
        let (x, y) = self.idx_to_pos(idx, grid_width);
        self.in_energy_zone(x, y)
    }
}

/// Per-program energy state
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProgramEnergy {
    /// Reserve energy epochs remaining (0 = no reserve)
    pub reserve_epochs: u32,
    /// Epochs since last copy interaction
    pub epochs_since_interaction: u32,
    /// Whether this program's tape is dead (zeroed)
    pub is_dead: bool,
    /// Whether this program was in an energy zone last epoch
    /// (used to detect zone transitions)
    was_in_zone: bool,
}

impl ProgramEnergy {
    /// Create a new program energy state (starts alive, no reserve)
    pub fn new() -> Self {
        Self {
            reserve_epochs: 0,
            epochs_since_interaction: 0,
            is_dead: false,
            was_in_zone: true, // Assume starting in zone
        }
    }

    /// Check if this program can mutate
    pub fn can_mutate(&self, in_zone: bool) -> bool {
        !self.is_dead && (in_zone || self.reserve_epochs > 0)
    }

    /// Check if this program is considered "alive" (not dead)
    pub fn is_alive(&self) -> bool {
        !self.is_dead
    }

    /// Record that a copy interaction occurred (resets interaction timer)
    pub fn record_interaction(&mut self) {
        self.epochs_since_interaction = 0;
    }

    /// Inherit energy from a copier
    ///
    /// - If copier is in zone, recipient gets max reserve
    /// - If copier is outside zone, recipient gets copier's remaining reserve
    /// - Resets interaction timer and marks as alive
    pub fn inherit_energy(&mut self, copier_reserve: u32, copier_in_zone: bool, max_reserve: u32) {
        self.is_dead = false;
        self.epochs_since_interaction = 0;
        self.reserve_epochs = if copier_in_zone {
            max_reserve
        } else {
            copier_reserve
        };
    }

    /// Update state for a new epoch
    ///
    /// Returns `true` if the program just died this epoch (tape should be zeroed)
    pub fn update_epoch(&mut self, in_zone: bool, config: &EnergyConfig) -> bool {
        if !config.enabled {
            return false; // Energy system disabled, no state changes
        }

        // Track zone transitions
        let just_left_zone = self.was_in_zone && !in_zone;
        let _just_entered_zone = !self.was_in_zone && in_zone;
        self.was_in_zone = in_zone;

        if in_zone {
            // In energy zone: reset to full energy, clear interaction timer
            self.reserve_epochs = config.reserve_duration;
            self.epochs_since_interaction = 0;
            // Note: dead programs in zone stay dead until they receive a copy
            return false;
        }

        // Outside energy zone
        if just_left_zone {
            // Just left zone: grant full reserve
            self.reserve_epochs = config.reserve_duration;
        } else if self.reserve_epochs > 0 {
            // Decrement reserve
            self.reserve_epochs -= 1;
        }

        // Increment interaction timer (if not dead)
        if !self.is_dead {
            self.epochs_since_interaction = self.epochs_since_interaction.saturating_add(1);

            // Check for death (interaction_death = 0 means infinite, never dies from timeout)
            if config.interaction_death > 0 && self.epochs_since_interaction > config.interaction_death {
                self.is_dead = true;
                return true; // Signal that tape should be zeroed
            }
        }

        false
    }

    /// Force death (zero out tape)
    pub fn kill(&mut self) {
        self.is_dead = true;
        self.reserve_epochs = 0;
    }

    /// Revive from death (called when receiving a copy)
    pub fn revive(&mut self, reserve: u32) {
        self.is_dead = false;
        self.reserve_epochs = reserve;
        self.epochs_since_interaction = 0;
    }
}

/// A copy event that occurred during BFF execution
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopyEvent {
    /// Program index that data was read FROM
    pub source_program: usize,
    /// Program index that data was written TO
    pub dest_program: usize,
}

impl CopyEvent {
    pub fn new(source_program: usize, dest_program: usize) -> Self {
        Self {
            source_program,
            dest_program,
        }
    }
}

/// Manages energy state for all programs in the simulation
#[derive(Clone, Debug)]
pub struct EnergySystem {
    /// Configuration
    pub config: EnergyConfig,
    /// Per-program energy states
    pub states: Vec<ProgramEnergy>,
    /// Grid dimensions for position calculations
    pub grid_width: usize,
    pub grid_height: usize,
}

impl EnergySystem {
    /// Create a new energy system
    pub fn new(config: EnergyConfig, grid_width: usize, grid_height: usize) -> Self {
        let num_programs = grid_width * grid_height;
        let mut states = vec![ProgramEnergy::new(); num_programs];

        // Initialize was_in_zone based on actual positions
        for (idx, state) in states.iter_mut().enumerate() {
            let x = idx % grid_width;
            let y = idx / grid_width;
            state.was_in_zone = config.in_energy_zone(x, y);
        }

        Self {
            config,
            states,
            grid_width,
            grid_height,
        }
    }

    /// Create a disabled energy system (all programs can always mutate)
    pub fn disabled(grid_width: usize, grid_height: usize) -> Self {
        Self::new(EnergyConfig::disabled(), grid_width, grid_height)
    }

    /// Check if the energy system is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the grid position from a program index
    pub fn idx_to_pos(&self, idx: usize) -> (usize, usize) {
        let x = idx % self.grid_width;
        let y = idx / self.grid_width;
        (x, y)
    }

    /// Check if a program is in an energy zone
    pub fn in_energy_zone(&self, idx: usize) -> bool {
        let (x, y) = self.idx_to_pos(idx);
        self.config.in_energy_zone(x, y)
    }

    /// Check if a program can mutate
    pub fn can_mutate(&self, idx: usize) -> bool {
        if !self.config.enabled {
            return true;
        }
        let in_zone = self.in_energy_zone(idx);
        self.states[idx].can_mutate(in_zone)
    }

    /// Check if a program is dead
    pub fn is_dead(&self, idx: usize) -> bool {
        self.states[idx].is_dead
    }

    /// Update all program states for a new epoch (call before pairing)
    ///
    /// Returns a list of program indices that just died (need tape zeroing)
    pub fn update_epoch(&mut self) -> Vec<usize> {
        let mut newly_dead = Vec::new();

        for idx in 0..self.states.len() {
            let (x, y) = self.idx_to_pos(idx);
            let in_zone = self.config.in_energy_zone(x, y);

            if self.states[idx].update_epoch(in_zone, &self.config) {
                newly_dead.push(idx);
            }
        }

        newly_dead
    }

    /// Process copy events from BFF execution (call after execution)
    ///
    /// Updates energy states based on copy operations:
    /// - Source records interaction (resets timer)
    /// - Destination inherits energy from source
    pub fn process_copy_events(&mut self, events: &[CopyEvent]) {
        for event in events {
            let source_state = self.states[event.source_program];
            let source_in_zone = self.in_energy_zone(event.source_program);

            // Source records interaction
            self.states[event.source_program].record_interaction();

            // Destination inherits energy
            self.states[event.dest_program].inherit_energy(
                source_state.reserve_epochs,
                source_in_zone,
                self.config.reserve_duration,
            );
        }
    }

    /// Get the energy state for a program
    pub fn get_state(&self, idx: usize) -> &ProgramEnergy {
        &self.states[idx]
    }

    /// Get mutable energy state for a program
    pub fn get_state_mut(&mut self, idx: usize) -> &mut ProgramEnergy {
        &mut self.states[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_source_contains() {
        let source = EnergySource::new(50, 50, 10);

        // Center is in range
        assert!(source.contains(50, 50));

        // Points within radius
        assert!(source.contains(55, 50)); // 5 away
        assert!(source.contains(50, 55)); // 5 away
        assert!(source.contains(57, 57)); // ~9.9 away

        // Points at exactly radius distance
        assert!(source.contains(60, 50)); // exactly 10 away

        // Points outside radius
        assert!(!source.contains(61, 50)); // 11 away
        assert!(!source.contains(50, 61)); // 11 away
        assert!(!source.contains(58, 58)); // ~11.3 away
    }

    #[test]
    fn test_energy_config_corners() {
        let config = EnergyConfig::corners(512, 256, 64);

        assert!(config.enabled);
        assert_eq!(config.sources.len(), 4);

        // Check corner positions
        assert_eq!(config.sources[0], EnergySource::new(64, 64, 64)); // Top-left
        assert_eq!(config.sources[1], EnergySource::new(447, 64, 64)); // Top-right
        assert_eq!(config.sources[2], EnergySource::new(64, 191, 64)); // Bottom-left
        assert_eq!(config.sources[3], EnergySource::new(447, 191, 64)); // Bottom-right
    }

    #[test]
    fn test_energy_config_in_zone() {
        let config = EnergyConfig::corners(512, 256, 64);

        // Points near corners should be in zone
        assert!(config.in_energy_zone(64, 64)); // At top-left source
        assert!(config.in_energy_zone(70, 70)); // Near top-left
        assert!(config.in_energy_zone(447, 191)); // At bottom-right source

        // Points in the middle should be outside
        assert!(!config.in_energy_zone(256, 128)); // Center of grid
        assert!(!config.in_energy_zone(200, 100)); // Random middle point
    }

    #[test]
    fn test_program_energy_can_mutate() {
        let mut state = ProgramEnergy::new();

        // In zone: can always mutate
        assert!(state.can_mutate(true));

        // Outside zone with no reserve: cannot mutate
        state.reserve_epochs = 0;
        assert!(!state.can_mutate(false));

        // Outside zone with reserve: can mutate
        state.reserve_epochs = 3;
        assert!(state.can_mutate(false));

        // Dead: cannot mutate even in zone
        state.is_dead = true;
        assert!(!state.can_mutate(true));
    }

    // #[test]
    // fn test_program_energy_leave_zone() {
    //     let config = EnergyConfig {
    //         enabled: true,
    //         sources: vec![EnergySource::new(50, 50, 10)],
    //         reserve_duration: 5,
    //         interaction_death: 10,
    //     };

    //     let mut state = ProgramEnergy::new();
    //     state.was_in_zone = true;

    //     // First epoch: still in zone
    //     state.update_epoch(true, &config);
    //     assert_eq!(state.reserve_epochs, 5);

    //     // Leave zone: should get full reserve
    //     state.update_epoch(false, &config);
    //     assert_eq!(state.reserve_epochs, 5);
    //     assert_eq!(state.epochs_since_interaction, 1);

    //     // Continue outside: reserve decrements
    //     state.update_epoch(false, &config);
    //     assert_eq!(state.reserve_epochs, 4);
    //     assert_eq!(state.epochs_since_interaction, 2);
    // }

    // #[test]
    // fn test_program_energy_death_timer() {
    //     let config = EnergyConfig {
    //         enabled: true,
    //         sources: vec![EnergySource::new(50, 50, 10)],
    //         reserve_duration: 5,
    //         interaction_death: 10,
    //     };

    //     let mut state = ProgramEnergy::new();
    //     state.was_in_zone = false;
    //     state.reserve_epochs = 0;

    //     // Simulate 10 epochs without interaction
    //     for i in 1..=10 {
    //         let died = state.update_epoch(false, &config);
    //         assert!(!died, "Should not die on epoch {}", i);
    //         assert_eq!(state.epochs_since_interaction, i as u8);
    //     }

    //     // 11th epoch: should die
    //     let died = state.update_epoch(false, &config);
    //     assert!(died);
    //     assert!(state.is_dead);
    // }

    // #[test]
    // fn test_program_energy_interaction_resets_timer() {
    //     let config = EnergyConfig {
    //         enabled: true,
    //         sources: vec![EnergySource::new(50, 50, 10)],
    //         reserve_duration: 5,
    //         interaction_death: 10,
    //     };

    //     let mut state = ProgramEnergy::new();
    //     state.was_in_zone = false;
    //     state.reserve_epochs = 0;

    //     // Simulate 5 epochs
    //     for _ in 0..5 {
    //         state.update_epoch(false, &config);
    //     }
    //     assert_eq!(state.epochs_since_interaction, 5);

    //     // Record interaction
    //     state.record_interaction();
    //     assert_eq!(state.epochs_since_interaction, 0);

    //     // Should survive another 10 epochs now
    //     for _ in 0..10 {
    //         let died = state.update_epoch(false, &config);
    //         assert!(!died);
    //     }
    // }

    #[test]
    fn test_program_energy_inherit_from_zone() {
        let mut state = ProgramEnergy::new();
        state.is_dead = true;

        // Inherit from a program in the energy zone
        state.inherit_energy(3, true, 5);

        assert!(!state.is_dead);
        assert_eq!(state.reserve_epochs, 5); // Gets max reserve
        assert_eq!(state.epochs_since_interaction, 0);
    }

    #[test]
    fn test_program_energy_inherit_from_outside() {
        let mut state = ProgramEnergy::new();
        state.is_dead = true;

        // Inherit from a program outside with 3 reserve
        state.inherit_energy(3, false, 5);

        assert!(!state.is_dead);
        assert_eq!(state.reserve_epochs, 3); // Gets copier's reserve
        assert_eq!(state.epochs_since_interaction, 0);
    }

    #[test]
    fn test_energy_system_creation() {
        let config = EnergyConfig::corners(100, 100, 20);
        let system = EnergySystem::new(config, 100, 100);

        assert_eq!(system.states.len(), 10000);
        assert!(system.is_enabled());
    }

    #[test]
    fn test_energy_system_update_epoch() {
        let config = EnergyConfig::custom(
            vec![EnergySource::new(5, 5, 3)],
            5,
            3, // Very short death timer for testing
        );
        let mut system = EnergySystem::new(config, 20, 20);

        // Program at (5, 5) is in zone
        let idx_in_zone = 5 + 5 * 20;
        assert!(system.in_energy_zone(idx_in_zone));

        // Program at (15, 15) is outside zone
        let idx_outside = 15 + 15 * 20;
        assert!(!system.in_energy_zone(idx_outside));

        // Update epochs until outside program dies
        for _ in 0..4 {
            let dead = system.update_epoch();
            // First few epochs: program has reserve from "leaving" zone
            // (actually started outside, but was_in_zone is initialized to true)
            assert!(dead.is_empty() || !dead.contains(&idx_in_zone));
        }
    }

    #[test]
    fn test_energy_system_process_copy_events() {
        let config = EnergyConfig::custom(
            vec![EnergySource::new(5, 5, 3)],
            5,
            10,
        );
        let mut system = EnergySystem::new(config, 20, 20);

        let idx_in_zone = 5 + 5 * 20;
        let idx_outside = 15 + 15 * 20;

        // Simulate a copy from in-zone to outside
        let events = vec![CopyEvent::new(idx_in_zone, idx_outside)];
        system.process_copy_events(&events);

        // Outside program should now have full reserve
        assert_eq!(system.states[idx_outside].reserve_epochs, 5);
        assert_eq!(system.states[idx_outside].epochs_since_interaction, 0);
    }

    #[test]
    fn test_energy_system_disabled() {
        let system = EnergySystem::disabled(100, 100);

        assert!(!system.is_enabled());

        // All programs can mutate when disabled
        for idx in 0..100 {
            assert!(system.can_mutate(idx));
        }
    }

    #[test]
    fn test_copy_event() {
        let event = CopyEvent::new(10, 20);
        assert_eq!(event.source_program, 10);
        assert_eq!(event.dest_program, 20);
    }
}
