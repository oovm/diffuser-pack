use super::*;
use serde::ser::SerializeStruct;

impl Display for ModelVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for ModelVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut ser = serializer.serialize_struct("Diffuser", 10)?;
        ser.serialize_field("version", self.as_str())?;
        ser.end()
    }
}

impl ModelVersion {
    fn as_str(&self) -> &'static str {
        match self {
            Self::V1_5 { .. } => "v1.5",
            Self::V2_1 { .. } => "v2.1",
            Self::XL { .. } => "XL",
            Self::XL_Turbo { .. } => "XL Turbo",
        }
    }
}
