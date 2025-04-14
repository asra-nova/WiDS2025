import sys
import gc
import numpy as np
import pandas as pd
import geomstats.backend as gs
import matplotlib.pyplot as plt
import geomstats.datasets.utils as data_utils
from tqdm import tqdm
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices

data_path = sys.argv[1]
mode = sys.argv[2]


df = pd.read_csv(data_path)


def process_single_row(atts):
    """
    Load brain connectome data and ADHD labels, returning symmetric matrices with ones on the diagonal.
    """
    atts = np.expand_dims(atts, axis=0)
    data = gs.array(atts)

    mat = SkewSymmetricMatrices(200).matrix_representation(data)
    mat = gs.eye(200) - gs.transpose(gs.tril(mat), (0, 2, 1))

    matrix = 1.0 / 2.0 * (mat + gs.transpose(mat, (0, 2, 1)))
    eigenvalues = np.linalg.eigvals(matrix)
    min_eigenvalue = np.min(eigenvalues)

    if min_eigenvalue < 0:
        correction = -min_eigenvalue + 1e-6
        correction_matrix = correction * np.eye(matrix.shape[0])
        return (matrix + correction_matrix).astype(np.float32).flatten()
    else:
        return matrix.astype(np.float32).flatten()


def load_connectomes_row_by_row_with_progress(df_conn, sub_fol, chunk_size=20):
    """
    Wrapper function to process brain connectome data row by row and return the resulting DataFrame.
    Each row is processed using the process_single_row function. A progress bar is displayed for row processing.

    Parameters:
    - df_conn (DataFrame): DataFrame containing the connectome data, with 200 nodes and 'participant_id'.

    Returns:
    - result_df (DataFrame): DataFrame containing the processed skew-symmetric matrices, one row for each participant.
    """
    # Initialize a list to store the processed matrices
    processed_matrices = []

    # Initialize a list to store the participant ids
    participant_ids = []

    # Iterate over each row in the DataFrame with tqdm for progress bar
    for idx, row in tqdm(
        df_conn.iterrows(), total=df_conn.shape[0], desc="Processing rows"
    ):
        # Extract the participant_id
        participant_id = row["participant_id"]
        participant_ids.append(participant_id)

        # Extract the edge attributes as a numpy array (excluding participant_id)
        edge_attributes = row.drop("participant_id").values.astype(np.float32)

        # Process the row using the process_single_row function
        processed_matrix = process_single_row(edge_attributes)

        # Append the processed matrix to the list
        processed_matrices.append(processed_matrix)

        # Every `chunk_size` rows, save and clear memory
        if (idx + 1) % chunk_size == 0:
            # Convert to DataFrame and store
            temp_df = pd.DataFrame(processed_matrices)
            temp_df["participant_id"] = participant_ids
            temp_df = temp_df[
                ["participant_id"]
                + [col for col in temp_df.columns if col != "participant_id"]
            ]

            # Optionally save to a file to reduce memory usage
            temp_df.to_csv(
                f"/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/{sub_fol}/skewsym_fnc/{idx // chunk_size}.csv",
                index=False,
            )

            # Clear lists and call garbage collection
            processed_matrices.clear()
            participant_ids.clear()
            gc.collect()

    # Convert the list of processed matrices into a DataFrame
    result_df = pd.DataFrame(processed_matrices)

    # Add the participant_id column back to the result DataFrame
    result_df["participant_id"] = participant_ids

    # Reorder the columns so that participant_id is the first column
    result_df = result_df[
        ["participant_id"]
        + [col for col in result_df.columns if col != "participant_id"]
    ]

    result_df.to_csv(
        f"/home/rmansouri1/school/advanced_ml/data/new/preprocessed_selected_features/{sub_fol}/skewsym_fnc/{(idx // chunk_size) + 1}.csv",
        index=False,
    )


load_connectomes_row_by_row_with_progress(df, mode)
