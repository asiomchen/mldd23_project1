import pandas as pd
import os
import time
def main():
    output_files = os.listdir('docking/outputs/')
    for file in output_files:
        if file.endswith('.csv') or file.startswith('chunk'):
            continue
        else:
            output_files.remove(file)

    concat_df = pd.read_csv(f'docking/outputs/{output_files[0]}')
    os.remove(f'docking/outputs/{output_files[0]}')

    for file in output_files[1:]:
        concat_df = pd.concat([concat_df, pd.read_csv(f'docking/outputs/{file}')])
        os.remove(f'docking/outputs/{file}')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    concat_df.to_csv(f'docking/outputs/concat_{timestamp}.csv', index=False)
    return

if __name__ == '__main__':
    main()