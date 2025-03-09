set -e

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
mv bazelisk-linux-amd64 /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

wget https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildifier-linux-amd64
mv buildifier-linux-amd64 /usr/local/bin/buildifier
chmod +x /usr/local/bin/buildifier
