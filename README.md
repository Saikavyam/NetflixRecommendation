Netflix Recommendation System



A content-based recommendation system that suggests similar movies and TV shows based on user input. Built using Python and Streamlit, this application leverages precomputed similarity matrices to provide quick and relevant recommendations.

Features



Content-Based Filtering: Recommends titles similar to the user's choice using content features.

Dual Dataset Support: Handles both movies and TV shows with separate datasets and similarity matrices.

Interactive UI: User-friendly interface built with Streamlit for seamless interaction.

Fast Recommendations: Utilizes precomputed similarity matrices for swift response times.


Installation



1. Clone the repository:
git clone https://github.com/Saikavyam/NetflixRecommendation.git
cd NetflixRecommendation

2.Install dependencies:
pip install -r requirements.txt

3.Run the application:
streamlit run app.py



Project Structure:

├── app.py                 # Main Streamlit application
├── movies_df.csv          # Dataset containing movie details
├── tv_show.csv            # Dataset containing TV show details
├── movies_sim.npz         # Precomputed similarity matrix for movies
├── tv_sim.npz             # Precomputed similarity matrix for TV shows
├── netflix-logo.json      # JSON file for the Netflix logo (used in UI)
├── requirements.txt       # List of Python dependencies
├── render.yaml            # Configuration for deployment (e.g., Render platform)
└── .devcontainer/         # Development container configuration (optional)

Datasets


1.movies_df.csv: Contains metadata for various movies, including titles, genres, and other relevant features.

2.tv_show.csv: Contains metadata for various TV shows, including titles, genres, and other relevant features.

Deployment


The application can be deployed on platforms like Render using the provided render.yaml configuration file.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

License

This project is licensed under the MIT License.

Acknowledgements

Streamlit for the interactive UI framework.

NumPy and Pandas for data manipulation and analysis.

Scikit-learn for machine learning utilities.
