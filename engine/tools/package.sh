#!/bin/zsh
# package.sh VERSION: build a self-contained release tarball, run from engine/ (or anywhere: it finds engine/).
#   -> release/inferd-VERSION-macosNN-arm64.tar.gz and its .sha256, unpacking to inferd-VERSION/ with everything needed
#      to run without a checkout: build/ (inferd, core.metallib, the release gate's tools), run.sh, install.sh,
#      requirements.txt, frontend/, tools/ (models.zsh, release_gate.sh, ref_logits.py, tie_check.py), README.md, LICENSE, NOTICE and
#      MLX's licence. No kernel sources, no engine sources, no third-party sources.
# NN is the macOS major the kernels were built for, read back from build/core.metallib and build/inferd. build.sh builds
# for the machine's own macOS and must stay that way: a lower Metal target made MLX's NAX matmuls compute WRONG logits
# on macOS 27 (2026-11-03). So this refuses MACOSX_DEPLOYMENT_TARGET, and a build whose target is not this macOS.
# Env: SKIP_BUILD=1 packages the existing build/ (default: ./build.sh first), OUT (default release/ here, gitignored).
emulate -L zsh
setopt no_unset pipe_fail err_exit
cd "${0:A:h}/.." || exit 1  # engine/
die() { print -u2 -r -- "package.sh: $*"; exit 1; }
VER=${1:-}
[[ $VER =~ '^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?$' ]] || die "usage: tools/package.sh VERSION (e.g. 0.3.0 or 0.3.0-rc1)"
[[ -z ${MACOSX_DEPLOYMENT_TARGET:-} ]] ||
  die "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET is set: never lower the kernels' target (see build.sh); unset it"
[[ $(uname -m) == arm64 ]] || die "build on Apple Silicon"
host=${$(sw_vers -productVersion)%%.*}

if [[ -z ${SKIP_BUILD:-} ]]; then
  print "package.sh: ./build.sh"
  ./build.sh || die "build.sh failed"
fi
bins=(inferd decode_test spec_test spec_sample_test grammar_test json_test spill_test weights_test)  # inferd + what release_gate.sh runs
for b in $bins; do [[ -x build/$b ]] || die "build/$b is missing: run ./build.sh"; done
[[ -f build/core.metallib ]] || die "build/core.metallib is missing: run ./build.sh"

# The target, from the files themselves
mtarget=$(xcrun metal-readobj --file-headers build/core.metallib 2>/dev/null | sed -n 's/^ *PlatformMajor: *//p' | head -1)
btarget=$(vtool -show-build build/inferd 2>/dev/null | sed -n 's/^ *minos *//p' | head -1)
btarget=${btarget%%.*}
[[ $mtarget == <-> && $btarget == <-> ]] || die "cannot read the build's macOS target (metallib '$mtarget', inferd '$btarget')"
[[ $mtarget == $btarget ]] || die "core.metallib targets macOS $mtarget but inferd targets macOS $btarget: rebuild"
[[ $mtarget == $host ]] ||
  die "the build targets macOS $mtarget on a macOS $host machine: build.sh builds for the machine's own macOS; rebuild here"
NN=$mtarget

# Provenance
commit=$(git rev-parse HEAD 2>/dev/null) || commit=unknown
dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')
(( dirty == 0 )) || print -u2 "package.sh: warning: $dirty tracked file(s) modified since $commit; BUILD-INFO records it"
NAME=inferd-$VER-macos$NN-arm64
OUT=${OUT:-$PWD/release}
mkdir -p -- "$OUT" || die "cannot create $OUT"
[[ ! -e $OUT/$NAME.tar.gz ]] || die "$OUT/$NAME.tar.gz exists; move it away first"

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/inferd-package.XXXXXX") || die "mktemp failed"
trap '[[ -d $STAGE && $STAGE == */inferd-package.* ]] && rm -rf -- "$STAGE"' EXIT
D=$STAGE/inferd-$VER
mkdir -p $D/build $D/frontend $D/tools $D/third_party/mlx
for b in $bins; do cp -p build/$b $D/build/; done
cp -p build/core.metallib $D/build/
cp -p run.sh install.sh requirements.txt $D/
cp -p tools/models.zsh tools/release_gate.sh tools/ref_logits.py tools/tie_check.py tools/test_ref_logits.py $D/tools/
cp -p frontend/serve.py frontend/structured.py frontend/vision.py frontend/image_decode.py $D/frontend/
cp -p frontend/test_hardening.py frontend/test_structured.py frontend/test_tools.py frontend/test_stream.py \
  frontend/test_image_decode.py frontend/test_vision.py $D/frontend/  # run by release_gate.sh
cp -p ../LICENSE $D/LICENSE
cp -p third_party/mlx/LICENSE third_party/mlx/README $D/third_party/mlx/
{
  cat ../NOTICE
  print "\nIn this binary distribution the MLX kernels are compiled into build/core.metallib; MLX's licence is"
  print "third_party/mlx/LICENSE and the vendored version and patches are listed in third_party/mlx/README."
} > $D/NOTICE
{
  print "version $VER"
  print "commit $commit"
  print "modified_files $dirty"
  print "macos_target $NN"
  print "built_on macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(xcodebuild -version 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')"
  print "built_at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  print "sha256 inferd $(shasum -a 256 < build/inferd | cut -d' ' -f1)"
  print "sha256 core.metallib $(shasum -a 256 < build/core.metallib | cut -d' ' -f1)"
} > $D/build/BUILD-INFO
cat > $D/README.md <<EOF
# inferd $VER for macOS $NN (Apple Silicon)

A native inference engine for many AI agents at once, with an OpenAI-compatible API. Built for **macOS $NN only**:
on another macOS use the matching download or build from source (https://github.com/Incept5-AI/inferd).

\`\`\`bash
./install.sh --model 27b          # Python 3.12 venv in ~/.venvs/inferd, then Qwen3.8-27B 4-bit + its drafter into ~/models
./run.sh start --model 27b        # sized from this Mac's memory; --dry-run prints the command lines; ./run.sh --help
curl localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}]}'
./run.sh stop 8300
\`\`\`

Other models: \`--model 27b-8bit\` (64 GB+), \`--model flash-next\` (Mac Studio, 256 GB), or \`--model /path/to/checkpoint\`
with \`--draft /path/to/drafter\`. Optional check of this build on your Mac (a few minutes, no other model running):
\`PY=~/.venvs/inferd/bin/python tools/release_gate.sh ~/models/Qwen3.8-27B-4bit ~/models/Qwen3.8-27B-DFlash2\`
(its logits reference uses mlx-lm if the venv has it, \`~/.venvs/inferd/bin/pip install mlx-lm\`, else mlx-vlm).

\`build/BUILD-INFO\` records the commit, the macOS target and the binaries' hashes. Licence: Apache 2.0 (LICENSE);
third-party notices: NOTICE.
EOF
chmod 755 $D/run.sh $D/install.sh $D/tools/release_gate.sh

# Self-containment: every file run.sh, install.sh and the gate use is present; no sources slipped in
for f in build/inferd build/core.metallib build/BUILD-INFO run.sh install.sh requirements.txt tools/models.zsh \
         tools/release_gate.sh tools/ref_logits.py tools/tie_check.py frontend/serve.py frontend/structured.py frontend/vision.py frontend/image_decode.py \
         LICENSE NOTICE README.md third_party/mlx/LICENSE; do
  [[ -f $D/$f ]] || die "staging is missing $f"
done
# ... and every binary, tool and front-end test release_gate.sh names (the gate grew after this list was written once)
for f in ${(u)${(f)"$(grep -o -E 'build/[a-z_0-9]+|tools/[a-z_0-9]+\.py' tools/release_gate.sh)"}} \
         ${(u)${(f)"$(sed -n 's/^for t in \(test_[a-z_ ]*\); do$/\1/p' tools/release_gate.sh | tr ' ' '\n' | sed 's|^|frontend/|; s|$|.py|')"}}; do
  [[ $f == build/core ]] && f=build/core.metallib
  [[ -f $D/$f ]] || die "staging is missing $f (used by release_gate.sh)"
done
srcs=($D/**/*.(metal|h|mm|cc|cpp|air)(N))
(( $#srcs == 0 )) || die "sources in the staging directory: ${srcs#$STAGE/}"

(cd $STAGE && COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --uname root --gname wheel -czf "$OUT/$NAME.tar.gz" inferd-$VER) ||
  die "tar failed"
(cd $OUT && shasum -a 256 $NAME.tar.gz > $NAME.tar.gz.sha256)
print "package.sh: $OUT/$NAME.tar.gz ($(du -h $OUT/$NAME.tar.gz | cut -f1), $(tar tzf $OUT/$NAME.tar.gz | grep -vc '/$') files)"
print "package.sh: $OUT/$NAME.tar.gz.sha256: $(cut -d' ' -f1 < $OUT/$NAME.tar.gz.sha256)"
