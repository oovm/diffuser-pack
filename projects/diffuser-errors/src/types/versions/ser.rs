use super::*;

impl Display for ModelVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1_5 => { f.write_str("v1.5") }
            Self::V2_1 => { f.write_str("v2.1") }
            Self::XL => { f.write_str("XL") }
            Self::XL_Turbo => { f.write_str("XL Turbo") }
        }
    }
}

impl Serialize for ModelVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        self.to_string().serialize(serializer)
    }
}
