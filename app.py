import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


class ChurnPredictionApp:
    def __init__(self):
        self.model, self.label_encoder_gender, self.onehot_encoder_geo, self.scaler = self.load_resources()

    def load_resources(self):
        """Load model, encoders, and scaler."""
        try:
            model = load_model('model.h5', compile=False)
        except Exception as e:
            st.error(f"Error while loading model: {e}")
            return None, None, None, None
        
        try:
            with open('label_encoder_gender.pkl', 'rb') as file:
                label_encoder_gender = pickle.load(file)
            
            with open('onehot_encoder_geo.pkl', 'rb') as file:
                onehot_encoder_geo = pickle.load(file)

            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        except Exception as e:
            st.error(f"Error while loading encoders or scaler: {e}")
            return model, None, None, None

        return model, label_encoder_gender, onehot_encoder_geo, scaler

    def prepare_input_data(self, user_inputs):
        """Prepare the input data for prediction."""
        geography = user_inputs['Geography']
        gender = user_inputs['Gender']
        input_data = pd.DataFrame({
            'CreditScore': [user_inputs['CreditScore']],
            'Gender': [self.label_encoder_gender.transform([gender])[0]],
            'Age': [user_inputs['Age']],
            'Tenure': [user_inputs['Tenure']],
            'Balance': [user_inputs['Balance']],
            'NumOfProducts': [user_inputs['NumOfProducts']],
            'HasCrCard': [user_inputs['HasCrCard']],
            'IsActiveMember': [user_inputs['IsActiveMember']],
            'EstimatedSalary': [user_inputs['EstimatedSalary']]
        })

        geo_encoded = self.onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=self.onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data, geo_encoded_df], axis=1)

        return self.scaler.transform(input_data)

    def display_result(self, prediction_proba):
        """Display the prediction result on Streamlit."""
        st.write(f'Churn Probability: {prediction_proba:.2f}')
        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')

    def run(self):
        """Run the Streamlit app."""
        st.title('Customer Churn Prediction')
        
        user_inputs = {
            'Geography': st.selectbox('Geography', self.onehot_encoder_geo.categories_[0]),
            'Gender': st.selectbox('Gender', self.label_encoder_gender.classes_),
            'Age': st.slider('Age', 18, 92),
            'Balance': st.number_input('Balance'),
            'CreditScore': st.number_input('Credit Score'),
            'EstimatedSalary': st.number_input('Estimated Salary'),
            'Tenure': st.slider('Tenure', 0, 10),
            'NumOfProducts': st.slider('Number of Products', 1, 4),
            'HasCrCard': st.selectbox('Has Credit Card', [0, 1]),
            'IsActiveMember': st.selectbox('Is Active Member', [0, 1])
        }

        if self.model and self.label_encoder_gender and self.onehot_encoder_geo and self.scaler:
            input_data_scaled = self.prepare_input_data(user_inputs)
            prediction = self.model.predict(input_data_scaled)
            self.display_result(prediction[0][0])
        else:
            st.error("Failed to load resources, please check the logs.")


# Run the app
if __name__ == '__main__':
    app = ChurnPredictionApp()
    app.run()
