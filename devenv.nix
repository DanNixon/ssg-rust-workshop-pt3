{ pkgs, ... }: {
  packages = with pkgs; [
    # Rust toolchain
    rustup

    # Build toolchain for hdf5
    cmake

    # Python stuff
    maturin
    python313
    python313Packages.virtualenv
  ];
}
