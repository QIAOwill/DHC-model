## DHC Model Introduction
- To address the challenge of predicting intercity mobility flows during holidays, we proposes a model named the `Dynamic Time-Space Holiday Feature Captor` *(**DHC**)*, as shown in the following figure. The model comprises four primary modules: `the Vision Dynamic Extension module` *(**VDE**)*, `the Spatiotemporal Feature Acquisition module` *(**SFA**)*, `the Vision Feature Extraction module` *(**VFE**)*, and `the Holiday Feature Fusion module` *(**HFF**)*.
- Specifically, the input data for the model includes **time series data** and **city attribute data**. Firstly, the VDE module captures an initial set of vision variables equal to the length of the prediction period. The SFA module aggregates the time series data along temporal and spatial dimensions using the initial vision variables and city attribute data. Based on the aggregation results and initial vision variables, the VFE module derives spatiotemporal feature information for each future time point. The HFF module then considers the similarities of analogous holidays to obtain trend information that includes holiday features for each future vision (future time point). Finally, the spatiotemporal feature information is fused with the trend information that includes holiday features, yielding the intercity mobility flow data for ***N*** cities for the next ***t*** days.

![Model architecture_02](https://github.com/user-attachments/assets/2970ac33-b3ec-45f1-a84b-760e02b179a8)

## How to run this project?
- The code of this project is divided into two parts: DTS and HFF **(the DTS part is the DHC model without HFF module)**. When the DHC model is run, it is necessary to run the HFF part first, and normalize the obtained result. The normalized result is used as the input data of the DTS part `[predict(MinMax).txt]`.
- Because of the confidentiality of the data, only the **data style** is given here, and the **data content is filled with "0"**.
### Data Preparation for DTS
- **Time series data** `[ OD Series.txt ]`: The intercity mobility flow between each city pair during 2023.01.01-2023.10.06 are recorded in detail.  
- **City attribute data** `[ data_Attribute.txt ]`: The result of city attribute data normalization.  
- **Train data** `[ Time Data.txt ]`: Time series data result after **normalization**.
- **Trend feature**`[ PCA.txt ]`: ***Trend Feature Extraction (green part)*** results.  
- **Holiday features**`[ predict(MinMax).txt ]`: The result of HFF part.  

### Packages Preparation for DTS
- Some Python packages need to be installed before running the code, such as **statsmodels, numba, pandas, torchsummary**(the code contains instructions to install these packages automatically). The **mmcv** can use "`pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html`" to install.
