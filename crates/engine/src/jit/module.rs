#[cfg(feature = "native")]
pub(super) use cranelift_jit::JITModule as CodeModule;

#[cfg(feature = "native")]
pub(super) fn new_module() -> Result<CodeModule, String> {
    let builder = cranelift_jit::JITBuilder::with_flags(&[], cranelift_module::default_libcall_names())
        .map_err(|error| error.to_string())?;
    Ok(CodeModule::new(builder))
}

#[cfg(not(feature = "native"))]
pub(super) use inert::{InertModule as CodeModule, new_module};

#[cfg(not(feature = "native"))]
mod inert {
    use cranelift_codegen::control::ControlPlane;
    use cranelift_codegen::isa::{self, OwnedTargetIsa, TargetIsa};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_codegen::{Context, ir};
    use cranelift_module::{
        DataDescription, DataId, FuncId, Linkage, Module, ModuleDeclarations, ModuleError,
        ModuleReloc, ModuleResult,
    };

    use crate::FxHashMap;

    pub(in crate::jit) struct InertModule {
        isa: OwnedTargetIsa,
        declarations: ModuleDeclarations,
        code: FxHashMap<FuncId, Box<[u8]>>,
    }

    pub(in crate::jit) fn new_module() -> Result<InertModule, String> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").map_err(|error| error.to_string())?;
        flags.set("is_pic", "false").map_err(|error| error.to_string())?;
        let isa = isa::lookup_by_name("aarch64-apple-darwin")
            .map_err(|error| error.to_string())?
            .finish(settings::Flags::new(flags))
            .map_err(|error| error.to_string())?;
        Ok(InertModule {
            isa,
            declarations: ModuleDeclarations::default(),
            code: FxHashMap::default(),
        })
    }

    impl InertModule {
        pub(in crate::jit) fn finalize_definitions(&mut self) -> ModuleResult<()> {
            Ok(())
        }

        pub(in crate::jit) fn get_finalized_function(&self, id: FuncId) -> *const u8 {
            self.code.get(&id).map_or(std::ptr::null(), |code| code.as_ptr())
        }
    }

    impl Module for InertModule {
        fn isa(&self) -> &dyn TargetIsa {
            &*self.isa
        }

        fn declarations(&self) -> &ModuleDeclarations {
            &self.declarations
        }

        fn declare_function(
            &mut self,
            name: &str,
            linkage: Linkage,
            signature: &ir::Signature,
        ) -> ModuleResult<FuncId> {
            let (id, _) = self.declarations.declare_function(name, linkage, signature)?;
            Ok(id)
        }

        fn declare_anonymous_function(&mut self, signature: &ir::Signature) -> ModuleResult<FuncId> {
            self.declarations.declare_anonymous_function(signature)
        }

        fn declare_data(
            &mut self,
            name: &str,
            linkage: Linkage,
            writable: bool,
            tls: bool,
        ) -> ModuleResult<DataId> {
            let (id, _) = self.declarations.declare_data(name, linkage, writable, tls)?;
            Ok(id)
        }

        fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> ModuleResult<DataId> {
            self.declarations.declare_anonymous_data(writable, tls)
        }

        fn define_function_with_control_plane(
            &mut self,
            func: FuncId,
            ctx: &mut Context,
            ctrl_plane: &mut ControlPlane,
        ) -> ModuleResult<()> {
            let compiled = ctx
                .compile(&*self.isa, ctrl_plane)
                .map_err(|error| ModuleError::Compilation(error.inner))?;
            self.code.insert(func, compiled.code_buffer().into());
            Ok(())
        }

        fn define_function_bytes(
            &mut self,
            func: FuncId,
            _alignment: u64,
            bytes: &[u8],
            _relocs: &[ModuleReloc],
        ) -> ModuleResult<()> {
            self.code.insert(func, bytes.into());
            Ok(())
        }

        fn define_data(&mut self, _data: DataId, _description: &DataDescription) -> ModuleResult<()> {
            Ok(())
        }
    }
}
