#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="data_mining/model"
TARGET_DIR="model"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

while IFS= read -r -d '' item; do
  name="$(basename "${item}")"
  dest="${TARGET_DIR}/${name}"

  if [[ -d "${item}" ]]; then
    rm -rf "${dest}"
    cp -a "${item}" "${dest}"
  else
    cp -a "${item}" "${dest}"
  fi
done < <(find "${SOURCE_DIR}" -mindepth 1 -maxdepth 1 -print0)

echo "Models copied from ${SOURCE_DIR} to ${TARGET_DIR}."
