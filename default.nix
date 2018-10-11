{ pkgs ? import <nixpkgs> {}}:
with pkgs;

{
  build = stdenv.mkDerivation rec {
    name = "image-classifier-${version}";
    version = "0.0.1";
    src = ./.;
    buildInputs = with python27Packages; [
      matplotlib
      numpy
      opencv3
      pillow
      scikitlearn
      scipy
      pylint
      ConfigArgParse
      makeWrapper
    ];
    installPhase = ''
      mkdir -p $out/bin
      cp -r main.py $out/bin/.main.py
      chmod +x $out/bin/.main.py
      makeWrapper $out/bin/.main.py $out/bin/image_classifier --prefix PYTHONPATH ':' $PYTHONPATH
    '';
  };
}
