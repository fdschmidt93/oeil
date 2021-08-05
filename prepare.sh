if ! [ -d "./data/summaries" ]
then
    mkdir -p ./data
    # tar -xzf ./observatory_summaries.tar.gz
    unzip ./observatory_summaries
    mv ./eu ./data/summaries
fi
