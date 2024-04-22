use super::*;



impl<'de> Deserialize<'de> for DiffuseTask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let mut config = DiffuseTask::default();
        deserializer.deserialize_any(DiffuserTaskVisitor { place: &mut config })?;
        Ok(config)
    }
    fn deserialize_in_place<D>(deserializer: D, place: &mut Self) -> Result<(), D::Error> where D: Deserializer<'de> {
        deserializer.deserialize_any(DiffuserTaskVisitor { place })
    }
}

pub struct DiffuserTaskVisitor<'i> {
    place: &'i mut DiffuseTask,
}

impl<'i, 'de> Visitor<'de> for DiffuserTaskVisitor<'i> {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        todo!()
    }
}