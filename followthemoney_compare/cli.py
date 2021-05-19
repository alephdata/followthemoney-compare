from pathlib import Path

import click
import pandas as pd
from followthemoney import model

from .lib.profiles import ProfileCollection
from .lib.utils import profiles_to_pairs_pandas, stdin_to_proxies
from .lib.word_frequency import WordFrequency, Frequencies
from . import models


MODEL_LOOKUP = {m.name: m for m in models.MODELS}


@click.group()
def main():
    pass


@main.command("create-data")
@click.argument("profile-dir", type=click.Path(file_okay=False))
@click.argument("output-file", type=click.File("wb+"))
def create_data(profile_dir, output_file):
    click.echo("Reading profiles", err=True)
    profiles = ProfileCollection.load_dir(profile_dir)
    click.echo("Transforming profiles to pandas pairs", err=True)
    df = profiles_to_pairs_pandas(profiles)
    click.echo("Serializing dataframe", err=True)
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
@click.option("--checkpoint-freq", default=100_000, type=int)
@click.option("--entities", type=click.File("r"), default="-")
@click.argument("output-dir", type=click.Path(file_okay=False, writable=True))
def create_word_frequency(
    entities, output_dir, confidence, error_rate, checkpoint_freq
):
    output_dir = Path(output_dir)

    document = model.get("Document")
    thing = model.get("Thing")
    try:
        proxies = stdin_to_proxies(
            entities, exclude_schema=[document], include_schema=[thing]
        )
        frequencies = Frequencies.from_proxies(
            proxies,
            confidence,
            error_rate,
            checkpoint_dir=output_dir,
            checkpoint_freq=checkpoint_freq,
        )
        frequencies.summarize()
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
