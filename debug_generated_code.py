
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Standardize column names to snake_case
    df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Cast columns correctly based on dtype hints
    df['32_502345269453031'] = pd.to_numeric(df['32_502345269453031'], errors='coerce')
    df['31_70700584656992'] = pd.to_numeric(df['31_70700584656992'], errors='coerce')
    
    return df

def make_visual_report(df: pd.DataFrame, out_png_path: str) -> None:
    sns.set_theme(style='whitegrid')
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='32_502345269453031', y='31_70700584656992', data=df)
    
    # Labeling axes and title
    plt.xlabel('Column 1 (numeric)')
    plt.ylabel('Column 2 (numeric)')
    plt.title('Scatter Plot of Numeric Columns')
    
    # Saving the plot to a high-DPI PNG file
    plt.savefig(out_png_path, dpi=300)
    
    # Closing the plot
    plt.close()