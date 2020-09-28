
#!/bin/bash
TARGETS=(
  "*.xml"
  "*.bin"
  "*.tgz"
  "*.whl"
  "*.tar.gz"
  "*.tflite"
  "*.pb"
  "*.pbtxt"
  "*.npy"
  "checkpoint"
  "*.data-00000-of-00001"
  "*.index"
  "*.meta"
)

target=$(printf " %s" "${TARGETS[@]}")
target=${target:1}

#clean git history

#sudo rm -rf .git/refs/original
#specific branch
#git filter-branch --index-filter "git rm -r --cached --ignore-unmatch ${target}" master..HEAD

#All branches
git filter-branch --index-filter "git rm -r --cached --ignore-unmatch ${target}" -- --all
