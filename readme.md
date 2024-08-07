# Towards Actionable Conversations: Enhancing User Engagement,
Understanding, and Trust with
Conversational Agents

*Streamlit has had an update which could improve the speed of this application. Future work could start by improving this*

This is the code for the thesis on the influence of conversational agents on non-expert users in the healthcare sector. To get started do the following steps:

1. ```
   git clone https://github.com/Vanherwegentim/HeartDiseaseRisk.git
   ```

2. ```
   pip install -r requirements.txt
   ```

3. ```
   streamlit run app.py
   ```

4. Access the application at  http://localhost:8501

The analytics are currently enabled and can be accessed at http://localhost:8501/?analytics=on. Scroll all the way down to view these. The code to connect these analytics to a cloud firestore database is currently commented by can be reconfigured to suit your needs.

An OpenAI api-key is needed to run the conversational agent. To add this to your application, you need to create a .streamlit folder. In that folder you need to create a secrets.toml file where you will put your api-key. This should look something like this:

```
OPENAI_API_KEY="sk-therestofyourkey"
PROD="False"
```

You would also need to the firestore credentials here if you wanted the analytics in the database.

The hosted version of this app can be found here https://heartdiseaseriskassessment.streamlit.app.

