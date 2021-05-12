from pathlib import Path

import click
import pandas as pd
from followthemoney import model

from .lib.profiles import ProfileCollection
from .lib.utils import profiles_to_pairs_pandas, stdin_to_proxies
from .lib.word_frequency import WordFrequency, word_frequency_from_proxies
from . import models


MODEL_LOOKUP = {m.name: m for m in models.MODELS}


@click.group()
def main():
    pass


@main.command("create-data")
@click.argument("profile-dir", type=click.Path(file_okay=False))
@click.argument("output-file", type=click.File("wb+"))
def create_data(profile_dir, output_file):
    profiles = ProfileCollection.load_dir(profile_dir)
    df = profiles_to_pairs_pandas(profiles)
    df.to_pickle(output_file)


@main.command("list-models")
def list_models():
    for model in MODEL_LOOKUP.keys():
        click.echo(model)


@main.command("train")
@click.option("--plot", default=None, type=click.Path(dir_okay=False, writable=True))
@click.argument("model_name", type=click.Choice(tuple(MODEL_LOOKUP.keys())))
@click.argument("data-file", type=click.File("rb"))
@click.argument("output-file", type=click.File("wb+"))
def train(model_name, data_file, output_file, plot=None):
    df = pd.read_pickle(data_file).query("weight > 0.1")
    model = MODEL_LOOKUP[model_name]
    m = model()
    df_train = df.query('phase == "train" and weight > 0.75').copy()
    print(f"Loaded {len(df)} samples")
    print(f"Training with {len(df_train)} samples")
    m.fit(df_train)
    print(m.summarize())
    output_file.write(m.evaluator.to_pickles())
    if plot:
        ax = m.precision_recall_accuracy_curve(df.query('phase == "valid"'))
        ax.figure.savefig(plot)


@main.command("create-word-frequency")
@click.option("--confidence", default=0.9995, type=float)
@click.option("--error-rate", default=0.0001, type=float)
@click.option("--entities", type=click.File("r"), default="-")
@click.option(
    "--document-frequency",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
)
@click.option(
    "--schema-frequency-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
)
@click.argument("output-file", type=click.Path(dir_okay=False, writable=True))
def create_word_frequency(
    entities,
    output_file,
    document_frequency,
    schema_frequency_dir,
    confidence,
    error_rate,
):
    schema_frequency_dir = Path(schema_frequency_dir)

    def save(wf, idf, sf):
        print("Saving Results")
        with open(output_file, "wb+") as fd:
            wf.save(fd)
        if idf:
            for w in idf.values():
                w.binarize()
            if len(idf) > 1:
                (root, *siblings) = list(idf.values())
                merged = root.merge(*siblings)
            else:
                (merged,) = idf.values()
            print("Binarized Document Frequency:")
            print(merged)
            with open(document_frequency, "wb+") as fd:
                merged.save(fd)
        if sf:
            schema_frequency_dir.mkdir(exist_ok=True, parents=True)
            for schema, frequency in sf.items():
                with open(schema_frequency_dir / f"{schema}.pro", "wb+") as fd:
                    frequency.save(fd)

    do_document_frequency = document_frequency is not None
    do_schema_frequency = schema_frequency_dir is not None
    document = model.get("Document")
    thing = model.get("Thing")
    try:
        proxies = stdin_to_proxies(
            entities, exclude_schema=[document], include_schema=[thing]
        )
        results = word_frequency_from_proxies(
            proxies,
            confidence,
            error_rate,
            document_frequency=do_document_frequency,
            schema_frequency=do_schema_frequency,
        )
        for wf, idf, sf in results:
            save(wf, idf, sf)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


@main.command("merge-word-frequency")
@click.option("--binarize", is_flag=True, default=False)
@click.argument(
    "input-word-frequencies",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True),
)
@click.argument("output-word-frequency", type=click.Path(dir_okay=False, writable=True))
def merge_word_frequency(input_word_frequencies, output_word_frequency, binarize):
    root = None
    for input_file in input_word_frequencies:
        with open(input_file, "rb") as fd:
            wf = WordFrequency.load(fd)
            if binarize:
                wf = wf.binarize()
            if root is None:
                root = wf
            else:
                root = root.merge(wf)
    with open(output_word_frequency, "wb+") as fd:
        root.save(fd)


if __name__ == "__main__":
    main()
