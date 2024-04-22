use super::*;

impl Serialize for DiffuseTask {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut ser = serializer.serialize_struct("DiffuseTask", 20)?;
        ser.serialize_field("positive-prompt", &self.prompt_positive)?;
        ser.serialize_field("negative-prompt", &self.prompt_negative)?;
        ser.serialize_field("height", &self.height)?;
        ser.serialize_field("width", &self.width)?;
        ser.end()
    }
}
