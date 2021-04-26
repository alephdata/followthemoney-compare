import click

import pandas as pd

from .lib.profiles import ProfileCollection
from .lib.utils import profiles_to_pairs_pandas
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
@click.argument("model_name", type=click.Choice(tuple(MODEL_LOOKUP.keys())))
@click.argument("data-file", type=click.File("rb"))
@click.argument("output-file", type=click.File("wb+"))
def train(model_name, data_file, output_file):
    df = pd.read_pickle(data_file)
    model = MODEL_LOOKUP[model_name]
    m = model(df)
    m.fit()
    output_file.write(m.evaluator.to_pickles())


if __name__ == "__main__":
    main()
