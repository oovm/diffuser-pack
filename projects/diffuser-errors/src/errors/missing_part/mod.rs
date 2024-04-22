use super::*;



impl From<MissingPartError> for DiffuserError {
    fn from(value: MissingPartError) -> Self {
        DiffuserErrorKind::MissingPart(value).into()
    }
}

impl MissingPartKind {
    pub fn with_local(self, local: PathBuf) -> MissingPartError {
        MissingPartError { part: self, local: Some(local), remote: None }
    }
    pub fn with_remote(self, remote: Url) -> MissingPartError {
        MissingPartError { part: self, local: None, remote: Some(remote) }
    }
}

impl MissingPartError {
    pub fn with_local(self, local: PathBuf) -> MissingPartError {
        Self { part: self.part, local: Some(local), remote: self.remote }
    }
    pub fn with_remote(self, remote: Url) -> MissingPartError {
        Self { part: self.part, local: self.local, remote: Some(remote) }
    }
}
