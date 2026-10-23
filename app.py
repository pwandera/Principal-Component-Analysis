import streamlit as st
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
sns.set_style('whitegrid')

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="PCA Showcase", layout="wide")

st.title("🌍 Country Data PCA Explorer")
st.caption("By: Paul Wandera")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        countries = pd.read_csv('countries/Country-data.csv', index_col='country')
        descriptions = pd.read_csv('countries/data-dictionary.csv')
        return countries, descriptions
    except FileNotFoundError:
        st.error("Error: Could not find the 'countries' folder or CSV files. Please ensure they are in the same directory.")
        st.stop()

countries, descriptions = load_data()
countries.columns = ['CHILD_MORTALITY', 'EXPORTS (%)', 'HEALTH (%)', 'IMPORTS (%)', 'INCOME ($)' ,  'INFLATION ($)', 'LIFE_EXPECTANCY', 'TOTAL_FERTILITY', 'GDP/CAPITA']

n, p = countries.shape

# --- MAIN LAYOUT TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Project Overview", 
    "Data Overview", 
    "Statistical Tests", 
    "PCA Results", 
    "PCA Visualizations"  # Renamed tab
])

# 1. Project Overview Tab
with tab1:
    st.header("About This Project")
    st.markdown("""
    Most real-world datasets come with a lot of correlated variables that cannot be easily visualized. Principal Component Analysis (PCA) fixes this issue by giving a high-dimensional dataset a faithful representation on a 2D or 3D plot.

    This project breaks down PCA method from scratch using step-by-step matrix algebra with NumPy, Pandas, and Matplotlib instead of calling scikit-learn in a few lines of code. The essence of PCA is to project the dataset on to a new set of axes (principal components) that maximize the variance in the original data while being orthogonal and uncorrelated to each other. These axes are the eigenvectors with the greatest eigenvalues of the covariance matrix. 
    """)

    st.info("""
    **📊 The Standardization Trap:** Working with a socio-economic dataset of 167 countries from Kaggle, the approach highlighted a classic data science trap: the importance of units and scales when working with data. Variables with large units (like income & GDP) can hijack the eigenvectors and give those variables overestimated weights on the new axes. After standardization of the variables and running an eigen-decomposition on the correlation matrix instead, this issue was resolved.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **✅ The Pros:** PCA is a powerful Machine Learning technique because it does not assume an underlying distribution of the data which hinders the inference of most techniques. PCA can be used for data reduction which makes other machine learning models faster. Using less variables is advantagious due to the Curse of Dimensionality.
        """)
    with col2:
        st.error("""
        **🚩 The Cons:** If the original variables are highly uncorrelated (the data cloud is somewhat spherical with respect to its dimension), projection onto lower dimensions can lose significant variance (information) and the interpretation of principal components is not always obvious or meaningful.
        """)

    st.warning("""
    **⚠️ Note on Eigenvectors:** One aspect to note is that the position on new axes is only important relative to other observations since the eigenvectors can be scaled by -1 and still be principal components.
    """)

    st.divider()
    st.header("The Algorithm")
    st.markdown(r"""
    Let $X \in \R^{n \times p}$ be your data matrix with $n$ rows and $p$ variables. The first step of PCA is to center our data.
    We define the mean vector as
    """)

    st.latex(r'\mu = (\frac{1}{n}) X^T \bf{1}_n')
    st.markdown('This will make each entry of $\mu$ corresponds to the mean of each column. Then we define $Q$ as ')
    st.latex(r'Q = \bf{1}_n \mu^T')
    st.markdown(r'where $Q$ is an $n \times p$ matrix and each column contains the repeated mean of a variable. Our centered matrix will therefore correspond to $X - Q$.')
    st.markdown('Next we compute the covariance and correlation matrices, $\Sigma$ and $R$ with')
    st.latex(r'\Sigma = \frac{1}{n} (X - Q)^T (X - Q)')
    st.latex(r'V = \text{diag}({\sqrt{(\Sigma_{ii}})})')
    st.markdown('where $V$ is a diagonal matrix that extracts the diagonal entries of $\Sigma$ and square roots them. Then, we find the standardised data matrix $Z$ by')
    st.latex(r'Z = (X - Q) V^{-1}')    
    st.latex(r'R = V^{-1} S V^{-1} = \frac{1}{n}Z^T Z')

    st.markdown('To find the directions of maximum variance in the standardized data, we can do spectal-decomposition on $R$:')
    st.latex(r'R = U \Lambda U^T')
    st.markdown(r'where $\Lambda \in \R^{p \times p}$ contains the eigenvalues that represent total variance along corresponding eigenvectors and $U \in \R^{p \times p}$ containts orthonormal eigenvectors so $U^T U = I$.')

    st.markdown('We then sort $\Lambda$ and $U$ respectively according to magnitude of eigenvalues and then out of the the $p$ columns of $U$ we shall pick $k$ to project the standardized data matrix onto.')
    st.markdown('Our output data matrix $Y$ is thus defined as:')
    st.latex(r'Y = Z U_k \in \R^{n \times k}')
    st.latex(r'k \le p')
    st.markdown(r"$y_{ij}$ will contain a linear combination of $z_{i}$ (the ith country's variables) weighted by $u_{j}$ (the jth eigenvector). In other words,")
    st.latex(r'y_{ij} = z_i^T u_j')

# 2. Data Overview Tab
with tab2:
    st.header("Raw Dataset")
    st.write(f"**Observations (Countries):** {n}")
    st.write(f"**Variables:** {p}")
    st.dataframe(countries)
    
    st.header("Variable Descriptions")
    st.dataframe(descriptions, hide_index=True)

    st.divider()
    st.header('Bivariate Plot of Columns', text_alignment="center")
    
    fig = sns.pairplot(countries, kind='scatter', corner=True, height=2.5, aspect = 1.2)
    st.set_page_config(layout="wide")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    

# --- MATH & COMPUTATIONS ---
X = countries.values

# Mean Vector
mu = (1/n) * X.T @ np.ones((n, 1))

# Covariance Matrix (S)
Q = np.ones((n, 1)) @ mu.T
S = (1/n) * (X - Q).T @ (X - Q)

# Correlation Matrix (R)
V = np.sqrt(np.diag(np.diag(S)))
V_inv = np.linalg.inv(V)
R = V_inv @ S @ V_inv

# Standardized Data Matrix
Z = (X - Q) @ V_inv

# Eigen-Decomposition
corr_eigen_values, corr_eigen_vectors = np.linalg.eigh(R)

# Sorting
sorted_indices = np.argsort(corr_eigen_values)[::-1]
corr_eigen_values = corr_eigen_values[sorted_indices]
corr_eigen_vectors = corr_eigen_vectors[:, sorted_indices]
corr_eigen_vectors[:, 0:2] *= -1

total_variance = sum(corr_eigen_values)
explained_variance = 100 * corr_eigen_values / total_variance

# Barlett's Test for Sphericity
alpha = 0.05

test_statistic = -(n - 1 - (2*p + 5)/6) * np.log(np.linalg.det(R))
p_value = 1 - scipy.stats.chi2.cdf(test_statistic, df = 0.5 * p * (p-1))

# 3. Matrix Computations Tab
with tab3:

    st.header("Bartlett's Test for Sphericity")
    st.markdown("""
        While the Principal Component Algorithm does not need any assumptions of your data following a multivariate distribution, if we make this assumption, we can determine if starting the PCA is actually beneficial.
        The premise of Maurice Bartlett's test for Sphericity is to check if the variables in your data matrix are actually correlated. This means that we are testing whether the correlation matrix of the data is actually an identity matrix with its off-diagonals (correlations) beiong 0s.
        
        Note: A statistical test is not actually necessary for this scenario since the dataset of countries is not a sample but a population. I still conducted this test just to showcase how it could supplement the analysis assuming the data were a sample.
    """)

    st.subheader("Assumptions of Bartlett's test:")

    st.text('- Multivariate Normality of the Data Matrix')
    st.text('- Continuous random variables')
    st.text('- Independence between samples')

    st.divider()
    st.subheader('Hypothesis Test:', text_alignment="center")
    st.latex(r'H_0 : R = I')
    st.latex(r'H_A : R \neq I')
    st.latex(r'\alpha = 0.05')
    st.text('\n')

    st.text('We assume the test statistic follows a Chi-Square Distribution with the following degrees of freedom being the amount of unique pairwise correlations:', text_alignment='center')

    st.latex(r' \chi^2_{ts} = - \left( n - 1 - \frac{2p + 5}{6} \right) \ln(\vert{}\hat{R}\vert{}) ')
    st.latex(r' df = \frac{p(p - 1)}{2} = 36')

    st.divider()
    st.markdown('Results:', text_alignment="center")

    st.latex(r'\chi^2_{ts} \approx 1169.737')
    st.latex(r' pvalue = \mathbb{P}[ \chi^2 \ge \chi^2_{ts} ] \approx 0')

    st.markdown('With a p-value of approximately 0, we can reject $H_0$ at all confidence levels. We can assume that the variables in our dataset are sufficiently correlated for PCA.')

# 4. PCA Results Tab
with tab4:
    st.header("Explained Variance")
    
    var_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(p)],
        "Explained Variance (%)": explained_variance,
        "Cumulative Variance (%)": np.round(np.cumsum(explained_variance), 2)
    })
    st.table(var_df.set_index("Principal Component"), )
    st.divider()

    # --- SIDE-BY-SIDE PLOTS ---
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Individual Scree Plot")
        # Creating the Scree Plot
        fig_scree, ax_scree = plt.subplots(figsize=(6, 5))
        
        # Plotting eigenvalues against their component numbers (1 to p)
        component_numbers = np.arange(1, p + 1)
        ax_scree.plot(component_numbers, corr_eigen_values, marker='o', linestyle='-', color='b')
        
        ax_scree.set_xlabel('Principal Component')
        ax_scree.set_ylabel('Eigenvalue (Variance)')
        ax_scree.set_xticks(component_numbers)
        ax_scree.grid(True)
        
        st.pyplot(fig_scree, use_container_width=False)
    
    with plot_col2:
        st.subheader("Cumulative Scree Plot")
        # Creating the Scree Plot
        fig_scree, ax_scree = plt.subplots(figsize=(6, 5))
        
        # Plotting eigenvalues against their component numbers (1 to p)
        component_numbers = np.arange(1, p + 1)
        ax_scree.plot(component_numbers, np.cumsum(corr_eigen_values), marker='o', linestyle='-', color='r')
        
        ax_scree.set_xlabel('Principal Component')
        ax_scree.set_ylabel('Cumulative Variance')
        ax_scree.set_xticks(component_numbers)
        ax_scree.grid(True)
        
        st.pyplot(fig_scree, use_container_width=False)
    
    st.write(f"**PC1, PC2 & PC3 % of Explained Variance:** {sum(explained_variance[0:3]):.3f}%")
    st.write(f"**PC1, PC2 & PC3 % of Lost Variance:** {100 - sum(explained_variance[0:3]):.3f}%")
    st.divider()

    st.header("Principal Component Equations (Top 3)", text_alignment='center')
    for i in range(3):
        equation = f"$PC_{i+1} = $ "
        terms = []
        for j in range(p):
            terms.append(f"$({corr_eigen_vectors[j][i]:.4f}) X{j+1}$")
        equation += " $+$ ".join(terms)
        st.markdown(equation, text_alignment='center')
    
    st.divider()

    st.header("Interpreting the Principal Components")
    st.markdown("The easy part of PCA is constructing these new axes. Interpreting them is usually the hard part. Based on the variable weights, the resulting 3 principal components can be interpreted as follows:")

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.subheader("🌍 PC1: Development Index")
        st.markdown("""
        * **$X_7$ (Life Expectancy):** 0.4258 
        * **$X_8$ (Total Fertility):** -0.4037 
        * **$X_1$ (Child Mortality):** -0.4195 
        * **$X_9$ (GDP per Capita):** 0.3926 
        * **$X_5$ (Income):** 0.3984
        
        We can interpret PC1 as a Development Index since weighting is focused on developmental factors. 
        Countries high in PC1 are likely developing countries with lower income, higher birth rates, and higher child mortalities.
        """)

    with col_b:
        st.subheader("🚢 PC2: Trade Volume Index")
        st.markdown("""
        * **$X_2$ (Imports):** 0.6132 
        * **$X_4$ (Exports):** 0.6718 
        
        We can interpret PC2 as a Global Trade Volume Index since weights are focused on imports and exports. 
        Countries high in PC2 are likely to be less active in global markets while countries low in PC2 are more engaged in global trade.
        """)

    with col_c:
        st.subheader("🏥 PC3: Health vs Inflation")
        st.markdown("""
        * **$X_3$ (Health Spending):** 0.5966  
        * **$X_6$ (Inflation):** -0.6132 
        
        Finally, we can interpret PC3 as the trade-off between health spending and inflation. 
        Countries high in PC3 are likely to have economies with governments spending relatively more on healthcare while keeping inflation in check. 
        Countries low in PC3 are likely to have economies that do not spend on healthcare but struggle to keep prices from rising.
        """)

    st.subheader('Weight Adjustment')
    st.markdown('To have PC1 reflect how developed a country is and have PC2 reflect how much global trade a country participates in, we multiply their eigenvectors by -1. This is not a problem since a negative eigenvector is still an eigenvector.')

# 5. PCA Visualizations Tab
with tab5:
    st.header("PCA Visualizations")
    
    # Controls
    st.subheader("3D Plot Settings")
    col1, col2 = st.columns([2, 1]) 
    with col1:
        sample_size = st.slider("Number of Countries to Sample for Plot", min_value=5, max_value=n, value=20, step=1)
    with col2:
        st.write("") 
        st.write("") 
        plot_all = st.checkbox("Plot All Countries")
    
    st.divider()
    
    # Projection Data Prep
    Y = Z @ corr_eigen_vectors[:, 0:3]
    pca_countries = pd.DataFrame(Y, index = countries.index, columns = ['PC1', 'PC2', 'PC3'])

    if plot_all:
        plot_data = pca_countries
    else:
        plot_data = pca_countries.sample(n=sample_size, random_state=1)
    
    st.write("**Dataframe of Projected Points (Top 3 PCs):**")
    st.dataframe(plot_data, use_container_width=True)

    st.divider()

    st.subheader("Sampled Data Projected on Top 3 PCs", text_alignment='center')
    
    spacer_left, plot_center, spacer_right = st.columns([1, 2, 1])
    
    with plot_center:
        # 3D Plotting
        fig_3d = plt.figure(figsize=(6, 5))
        ax_3d = fig_3d.add_subplot(projection='3d')

        ax_3d.scatter(xs=plot_data['PC1'], ys=plot_data['PC2'], zs=plot_data['PC3'], c='r', marker='o')
        ax_3d.set_xlabel(f'PC1: {explained_variance[0]:.1f}%')
        ax_3d.set_ylabel(f'PC2: {explained_variance[1]:.1f}%')
        ax_3d.set_zlabel(f'PC3: {explained_variance[2]:.1f}%')

        # Add text labels for the countries
        for i in range(len(plot_data)):
            ax_3d.text(plot_data.iloc[i]['PC1'], 
                    plot_data.iloc[i]['PC2'], 
                    plot_data.iloc[i]['PC3'], 
                    plot_data.index[i], 
                    size=7)

        st.pyplot(fig_3d, use_container_width=False)

    st.divider()
    st.subheader(r"Outlier Detection")
    st.markdown(r"If $S$ is a sample covariance matrix and $\bar{x}$ is a sample mean vector, and $\vec{x}$ is a multivariate observation, then the Mahalanobis distance $D_M$ is defined as:")
    st.latex(r'D_M(\vec{x}) = \sqrt{(\vec{x} - \bar{x})^T S^{-1} (\vec{x} - \bar{x})}')
    st.markdown(r'In the case of our 3-dimensional PCA outputs which are mean centered, $\bar{x} = \vec{0}$ and ')
    st.latex(r'''
    S = \begin{bmatrix}
    \lambda_1    & 0  & 0      \\
    0      & \lambda_2   & 0      \\
    0      & 0  & \lambda_3
    \end{bmatrix}
    ''')
    st.markdown('So,')
    st.latex(r'''
    S^{-1} = \begin{bmatrix}
    \frac{1}{\lambda_1}    & 0   & 0      \\
    0      & \frac{1}{\lambda_2}  & 0      \\
    0      & 0    & \frac{1}{\lambda_3}
    \end{bmatrix}
    ''')
    st.markdown('Thus,')
    st.latex(r'D_M(\vec{x}) = \sqrt{\frac{{x_1}^2}{\lambda_1} + \frac{{x_2}^2}{\lambda_2} + \frac{{x_3}^2}{\lambda_3}}')

    distances = []
    data = pca_countries.values
    lambdas = corr_eigen_values[0:3]

    for i in range(n):
        vector = data[i]
        d = 0

        for j in range(3):
            d += (vector[j]) ** 2 / lambdas[j]

        d **= 0.5
        distances.append(d)

    sorted_distances = np.argsort(distances)[::-1]
    pca_countries['MAHALANOBIS'] = distances
    sorted_pc1 = np.argsort(pca_countries['PC1'].values)

    st.divider()
    st.write("**Dataframe with Mahalanobis Distances**")
    st.dataframe(pca_countries.iloc[sorted_pc1[0:20], 0:4])
    st.divider()

    pairplot_data = pca_countries.iloc[:, 0:3]
    pairplot_data['Highlight'] = (pairplot_data.index == 'Nigeria')
    pairplot_data['Highlight'] = pairplot_data['Highlight'].map({
        True: 'Nigeria', 
        False: 'Other Countries'
    })

    pairplot_data.loc['Singapore', 'Highlight'] = 'Singapore'

    spacer_left, plot_center, spacer_right = st.columns([1, 2, 1])
    with plot_center:
        st.subheader('Bivariate Plot of Projected Data', text_alignment="center")
        fig = sns.pairplot(pairplot_data, kind='scatter', corner=True, hue='Highlight', palette = {'Other Countries': 'steelblue', 'Nigeria': 'red', 'Singapore': 'orange'})
        st.pyplot(fig)

    st.markdown('We can compare this plot with the original bivariate plots. We can see that our new variabels are centered at $0$ and, more importantly, they are approximately uncorrelated with each other. In the original plot we ' \
                'could not look at a plot between 2 variables without worrying about the effects of the other 7. However with the PCA outputs, we can do this since the eigenvectors we mapped our original data onto are ' \
                'orthogonal')

    st.divider()
    st.subheader("Key Takeaways")
    st.markdown('''
    _From this we can see two contrasting outliers. Nigeria is considered a multivariate outlier because despite having an average Trade Participation score ($PC_2$), it's Developmental ($PC_1$) and Health-Inflation ($PC_3$) scores are the weakest of the all countries. Singapore on the other hand is an outlier because its has the Best Trade Participation score while having the second best Development score and an average Health-Inflation score._

    _From the data, we may conclude that Nigeria may have underlying factors that prevent its Development from taking advantage of its Trade Participation. However, this conclusion is only as good as the data collected._

    *Link to dataset:* https://www.kaggle.com/datasets/vipulgohel/clustering-pca-assignment/data
    ''')