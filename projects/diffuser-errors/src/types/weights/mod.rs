use super::*;

#[derive(Debug)]
pub struct WeightInfo {
    pub remote: Url,
    pub local: String,
    pub r#type: DType,
    pub hash: u64,
}

impl Serialize for WeightInfo {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de> Deserialize<'de> for WeightInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}
struct WeightVisitor {
    data: WeightInfo,
}

impl<'i, 'de> Visitor<'de> for &'i mut WeightVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        formatter.write_str("expect a `WeightInfo` object")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.to_ascii_lowercase().as_str() {
                "id" => self.data.id = map.next_value()?,
                "data" => {
                    let str = map.next_value::<String>()?;
                    match DType::from_str(&str) {
                        Ok(o) => self.data.r#type = o,
                        Err(_) => Err(A::Error::custom("dot a valid data type"))?,
                    }
                }
                "remote" => self.data.remote = map.next_value()?,
                "hash" => self.data.hash = map.next_value()?,
                _ => {}
            }
        }
        Ok(())
    }
}
