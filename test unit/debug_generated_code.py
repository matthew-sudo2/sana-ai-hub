
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Standardize column names to snake_case
    df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Cast columns correctly based on dtype hints
    df['country'] = df['country'].astype(int)
    df['year'] = df['year'].astype(int)
    df['population'] = df['population'].astype(float)
    
    return df

def make_visual_report(df: pd.DataFrame, out_png_path: str) -> None:
    sns.set_theme(style='whitegrid')
    
    # Plotting population over years
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='year', y='population', data=df)
    plt.title('Population Over Years')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.savefig(out_png_path, dpi=300)
    plt.close()