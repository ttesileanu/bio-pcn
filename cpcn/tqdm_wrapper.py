"""Define a wrapper for tqdm that outputs useful training progress info."""

import numpy as np

from typing import Iterable

from tqdm.autonotebook import tqdm


def tqdmw(train_it: Iterable, tqdm=tqdm) -> Iterable:
    """Add a progress bar to a training iterable, outputting progress information.

    The iterable must have a `len`gth.
    
    :param train_it: iteratble to wrap
    :param tqdm: tqdm(-like) construct to use
    """
    pbar = tqdm(total=len(train_it))
    for batch in train_it:
        yield batch

        progress_info = {}
        if (
            hasattr(train_it, "trainer")
            and hasattr(train_it.trainer, "metrics")
            and hasattr(train_it.trainer, "history")
            and hasattr(train_it.trainer.history, "validation")
        ):
            validation = train_it.trainer.history.validation
            for metric in train_it.trainer.metrics:
                if not isinstance(metric, str):
                    continue

                # make this robust -- would prefer to not fail due to progress bar
                try:
                    last_val = float(validation[metric][-1])
                    progress_str = f"{last_val:.3g}"
                except:
                    progress_str = "???"

                progress_info["val " + metric] = progress_str

        pbar.set_postfix(progress_info, refresh=False)
        pbar.update()

    pbar.close()
