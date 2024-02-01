# script to use PorphyStruct to analyze corrole and porphyrin non-planarity
for file in data/curated/$1/*; do
    echo $file
    ./porphystruct/PorphyStruct.CLI analyze -x $file
    mv data/curated/$1/*.json data/nonplanarity/$1/
    rm data/curated/$1/*.md
done