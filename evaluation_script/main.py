import pandas as pd
import numpy as np
import sklearn.metrics as metrics

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Evaluation is performed calculating RMSE of prediction for each meter. Final score is -average RMSE,
    """

    print("Starting Evaluation Test.....")
    print(phase_codename)
    print(user_submission_file)
    print(test_annotation_file)

    try:

        print("entro in TRY")

        y_pred = pd.read_csv(user_submission_file, header=0, sep=';')
        y_true = pd.read_csv(test_annotation_file, header=0, sep=';')

        df = pd.merge(y_true, y_pred, how='left', on=['meter', 'date'], suffixes=("_true", "_pred")).sort_values(
            ['meter']).fillna(-1)

        contatori = list(df['meter'].drop_duplicates())
        rmse = list()

        for i in contatori:
            df_i = df.loc[df['meter'].isin([i])].sort_values(by=['date'])
            y_pred_i = df_i['value_pred']
            y_true_i = df_i['value_true']

            rmse.append(round(np.sqrt(metrics.mean_squared_error(y_pred_i, y_true_i)), 2))

        score = np.mean(rmse)


    except Exception:
        rmse = list()
        print("entro in EXCEPT")
        score = 998


    print(rmse)
    print(score)

    output = {}
    output['result'] = [
        {
            'all_data': {
                'RMSE': score,
                'Score': -score
            }
        }
    ]

    print(output)

    return output