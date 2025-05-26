{ pkgs, ... }: {
  packages = with pkgs; [
    # Rust toolchain
    rustup

    # Build toolchain for hdf5
    cmake
  ];
}
