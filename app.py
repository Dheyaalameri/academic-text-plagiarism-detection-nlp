import streamlit as st
import pandas as pd
import plagiarism_detection as pd_engine

st.set_page_config(page_title="Plagiarism Detector", layout="wide")

st.title("Academic Text Plagiarism Detection Using NLP")
st.markdown("### Detect textual similarity between student assignments")

st.markdown("---")
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Select Input Method:", ("Upload .txt Files", "Use Existing Data Folder"))

filenames = []
documents = []

if data_source == "Upload .txt Files":
    uploaded_files = st.file_uploader("Upload Student Assignments (.txt)", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            filenames.append(uploaded_file.name)
            # Read and decode the file content
            documents.append(uploaded_file.read().decode("utf-8"))
        st.info(f"Uploaded {len(filenames)} files.")

elif data_source == "Use Existing Data Folder":
    folder_path = st.sidebar.text_input("Data Folder Path", value="data")
    if st.sidebar.button("Load Files"):
        try:
            filenames, documents = pd_engine.load_data(folder_path)
            if filenames:
                st.success(f"Successfully loaded {len(filenames)} files from '{folder_path}'.")
                with st.expander("View Loaded Files"):
                    st.write(filenames)
            else:
                st.warning(f"No .txt files found in '{folder_path}'.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

st.markdown("---")

if st.button("Run Plagiarism Detection", type="primary"):
    if not documents or len(documents) < 2:
        st.error("Insufficient data. Please provide at least 2 documents to compare.")
    else:
        with st.spinner("Analyzing documents..."):
            try:
                # 1. Preprocessing
                processed_docs = [pd_engine.preprocess_text(doc) for doc in documents]

                # 2. Feature Extraction
                vectorizer, tfidf_matrix = pd_engine.extract_features(processed_docs)

                # 3. Similarity Calculation
                cosine_sim_matrix = pd_engine.calculate_cosine_similarity(tfidf_matrix)

                # 4. Generate Results
                results = []
                num_files = len(filenames)

                for i in range(num_files):
                    for j in range(i + 1, num_files):
                        file1 = filenames[i]
                        file2 = filenames[j]
                        
                        cosine_score = cosine_sim_matrix[i][j]
                        jaccard_score = pd_engine.calculate_jaccard_similarity(processed_docs[i], processed_docs[j])
                        verdict = pd_engine.classify_similarity(cosine_score, jaccard_score)

                        results.append({
                            "File A": file1,
                            "File B": file2,
                            "Cosine Similarity": cosine_score,
                            "Jaccard Similarity": jaccard_score,
                            "Verdict": verdict
                        })
                
                # 5. Display Results
                if results:
                    df_results = pd.DataFrame(results)
                    # Sort by Cosine Similarity descending
                    df_results = df_results.sort_values(by="Cosine Similarity", ascending=False)
                    
                    st.subheader("Plagiarism Detection Report")
                    
                    # Formatting for better display
                    st.dataframe(
                        df_results.style.format({
                            "Cosine Similarity": "{:.4f}",
                            "Jaccard Similarity": "{:.4f}"
                        }).applymap(lambda x: "color: red; font-weight: bold" if x == "High Similarity" else 
                                            ("color: orange" if x == "Moderate Similarity" else "color: green"), 
                                            subset=["Verdict"])
                    , use_container_width=True)
                    
                else:
                    st.info("No pairs to compare.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
