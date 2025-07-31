#!/bin/bash

data_ddg="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/data_csv_ddg.zip?inline=false"
data_frustra="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/data_csv_frustration.zip?inline=false"
data_pdb="https://git.iwe-lab.de/mabsi/frustrampnn_data/-/raw/main/pdb_files.zip?inline=false"

# Download data
zip_files=($data_ddg $data_frustra $data_pdb)
output_files=("data_csv_ddg.zip" "data_csv_frustration.zip" "pdb_files.zip")

for i in {0..2}
do
    if [ -f ${output_files[i]} ]; then
        echo "File ${output_files[i]} already exists. Skipping download."
        continue
    fi
    echo "Downloading ${output_files[i]}"
    curl -L -o ${output_files[i]} ${zip_files[i]}
    
    if [ ! -f ${output_files[i]} ]; then
        echo "Failed to download ${output_files[i]}. Exiting."
        continue
    fi

    echo "Unzipping ${output_files[i]}"
    unzip ${output_files[i]}

    echo "Removing ${output_files[i]}"
    rm ${output_files[i]}
done

# rm *.zip