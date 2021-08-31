if ! [ -d "./data/summaries" ]
then
    mkdir -p ./data
    # tar -xzf ./observatory_summaries.tar.gz
    unzip ./raw_data.zip
    mv ./summaries ./data/
fi
