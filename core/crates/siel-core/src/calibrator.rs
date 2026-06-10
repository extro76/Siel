use serde::{Deserialize, Serialize};

use crate::{Result, SielError};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Calibrator {
    Fixed,
    Isotonic(IsotonicCalibrator),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    pub version: u32,
    pub x_thresholds: Vec<f32>,
    pub y_probabilities: Vec<f32>,
}

impl IsotonicCalibrator {
    pub fn validate(&self) -> Result<()> {
        if self.x_thresholds.is_empty() || self.x_thresholds.len() != self.y_probabilities.len() {
            return Err(SielError::InvalidInput(
                "isotonic calibrator requires aligned non-empty arrays".to_string(),
            ));
        }

        if self.x_thresholds.windows(2).any(|w| w[0] > w[1]) {
            return Err(SielError::InvalidInput(
                "isotonic x_thresholds must be sorted ascending".to_string(),
            ));
        }

        if self
            .y_probabilities
            .iter()
            .any(|p| !(0.0..=1.0).contains(p))
        {
            return Err(SielError::InvalidInput(
                "isotonic probabilities must be in [0, 1]".to_string(),
            ));
        }

        Ok(())
    }

    pub fn predict(&self, raw_score: f32) -> Result<f32> {
        self.validate()?;
        let idx = match self
            .x_thresholds
            .binary_search_by(|probe| probe.total_cmp(&raw_score))
        {
            Ok(idx) => idx,
            Err(0) => 0,
            Err(idx) => idx - 1,
        };
        Ok(self.y_probabilities[idx])
    }
}

impl Calibrator {
    pub fn predict(&self, raw_score: f32) -> Result<(f32, bool)> {
        match self {
            Calibrator::Fixed => Ok((raw_score.clamp(0.0, 1.0), false)),
            Calibrator::Isotonic(model) => Ok((model.predict(raw_score)?.clamp(0.0, 1.0), true)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceProfile {
    pub profile_id: String,
    pub threshold_high: f32,
    pub threshold_low: f32,
    pub calibrator: Calibrator,
}

impl Default for ConfidenceProfile {
    fn default() -> Self {
        Self {
            profile_id: "default".to_string(),
            threshold_high: 0.82,
            threshold_low: 0.62,
            calibrator: Calibrator::Fixed,
        }
    }
}
