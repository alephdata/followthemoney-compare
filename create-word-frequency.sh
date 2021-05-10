gcs_bucket=$1

function collection-cat() {
    url=$1
    collection=$( basename $url .json )
    >&2 echo "Reading collection ${collection} from ${url}"
    gsutil cat $url | jq --compact-output --arg collection "$collection" '. + {collection: $collection}'
}
export -f collection-cat

gsutil ls "gs://${gcs_bucket}/**/*.json" | \
    grep -v "_profiles.json" | \
    xargs -P 1 -d '\n' -n 1 -I{} bash -c 'collection-cat "{}"' | \
    head -n 500000 | \
    pv -l | \
    ftm-compare create-word-frequency \
        --document-frequency data/word_freq/document_frequency.pro \
        --schema-frequency-dir data/word_freq/schema_frequency/ \
        data/word_freq/token_frequency.pro
