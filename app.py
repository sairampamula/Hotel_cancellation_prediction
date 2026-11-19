import datetime as dt
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "final_model.pkl",
]
SCHEMA_PATH = BASE_DIR / "input_schema.json"


# ---------------------------------------------------------------------
# Load model and optional input schema
# ---------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as file:
        return cloudpickle.load(file)


def load_schema():
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write(
    "Enter the booking details below to estimate whether the reservation "
    "is likely to be cancelled."
)

if model_path is None:
    st.error(
        "Model file not found. Place `final_model.pkl` or `best_model.pkl` "
        "in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Use the same Python environment and compatible package versions "
        "that were used to train and save the model."
    )
    st.stop()

schema = load_schema()

categorical_options = schema.get("categorical_options", {})
agent_options = [int(a) for a in schema.get("agent_options", [])]

# Fallback options are used only if input_schema.json is unavailable.
DEFAULT_OPTIONS = {
    "hotel": ["Resort Hotel", "City Hotel"],
    "meal": ["BB", "HB", "SC", "Undefined", "FB"],
    "market_segment": [
        "Online TA", "Offline TA/TO", "Groups", "Direct",
        "Corporate", "Complementary", "Aviation", "Undefined",
    ],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
    "reserved_room_type": list("ABCDEFGHLP"),
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "arrival_date_month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
}


DEFAULT_COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL",
                     "BRA", "NLD", "USA", "CHE", "Unknown"]


def choices_for(column):
    values = categorical_options.get(column) or DEFAULT_OPTIONS[column]
    # Keep the widget usable if a schema contains an unexpectedly large list.
    return [str(value) for value in values]


def select_value(column, default_index=0):
    options = choices_for(column)
    if not options:
        options = DEFAULT_OPTIONS[column]
    return st.selectbox(
        column.replace("_", " ").title(),
        options,
        index=min(default_index, len(options) - 1),
    )


# ---------------------------------------------------------------------
# Booking input form
# The names below match the raw columns expected by the trained pipeline.
# ---------------------------------------------------------------------
with st.form("booking_details"):
    st.subheader("1. Hotel and arrival details")
    c1, c2, c3 = st.columns(3)

    with c1:
        hotel = select_value("hotel")
        arrival = st.date_input(
            "Arrival Date",
            value=dt.date.today() + dt.timedelta(days=30),
            min_value=dt.date.today(),
        )

    with c2:
        market_segment = select_value("market_segment")
        distribution_channel = select_value("distribution_channel")

    with c3:
        reserved_room_type = select_value("reserved_room_type")
        meal = select_value("meal")

    # The four arrival columns the model expects, derived from one date
    MONTHS = DEFAULT_OPTIONS["arrival_date_month"]
    arrival_date_year = arrival.year
    arrival_date_month = MONTHS[arrival.month - 1]
    arrival_date_day_of_month = arrival.day
    arrival_date_week_number = arrival.isocalendar()[1]

    st.divider()
    st.subheader("2. Stay and guest details")
    c1, c2, c3 = st.columns(3)

    with c1:
        adults = st.number_input("Adults", min_value=0, max_value=20, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=10, value=0)

    with c2:
        stays_in_weekend_nights = st.number_input(
            "Weekend Nights", min_value=0, max_value=60, value=1
        )
        stays_in_week_nights = st.number_input(
            "Week Nights", min_value=0, max_value=60, value=2
        )
        adr = st.number_input(
            "Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0,
            value=100.0, step=5.0,
        )

    with c3:
        customer_type = select_value("customer_type")
        country_list = [str(c) for c in categorical_options.get("country", [])] or DEFAULT_COUNTRIES
        if "Unknown" not in country_list:
            country_list.append("Unknown")
        country = st.selectbox(
            "Country",
            country_list,
            index=country_list.index("PRT") if "PRT" in country_list else 0,
        )
        if agent_options:
            agent_choice = st.selectbox("Agent ID", ["Unknown"] + agent_options)
        else:
            agent_choice = st.number_input(
                "Agent ID (0 if unknown)", min_value=0, value=0, step=1
            )

    st.divider()
    st.subheader("3. Booking history and preferences")
    c1, c2, c3 = st.columns(3)

    with c1:
        is_repeated_guest = st.selectbox("Repeated Guest", [0, 1], format_func=lambda x: "Yes" if x else "No")
        previous_cancellations = st.number_input(
            "Previous Cancellations", min_value=0, value=0, step=1
        )
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Cancelled", min_value=0, value=0, step=1
        )

    with c2:
        booking_changes = st.number_input(
            "Booking Changes", min_value=0, value=0, step=1
        )
        deposit_type = select_value("deposit_type")
        required_car_parking_spaces = st.number_input(
            "Required Car Parking Spaces", min_value=0, max_value=10, value=0
        )

    with c3:
        total_of_special_requests = st.number_input(
            "Total Special Requests", min_value=0, max_value=10, value=0
        )
        lead_time = st.number_input(
            "Lead Time (days)", min_value=0, max_value=1000, value=30
        )
        days_in_waiting_list = st.number_input(
            "Days in Waiting List", min_value=0, max_value=5000, value=0
        )

    submitted = st.form_submit_button(
        "Predict Cancellation", type="primary", use_container_width=True
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if submitted:
    if adults + children + babies == 0:
        st.error("Please enter at least one guest.")
        st.stop()

    # Keep the raw feature names exactly as used during model training.
    input_data = {
        "hotel": hotel,
        "lead_time": int(lead_time),
        "days_in_waiting_list": int(days_in_waiting_list),
        "arrival_date_year": int(arrival_date_year),
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": int(arrival_date_week_number),
        "arrival_date_day_of_month": int(arrival_date_day_of_month),
        "stays_in_weekend_nights": int(stays_in_weekend_nights),
        "stays_in_week_nights": int(stays_in_week_nights),
        "adults": int(adults),
        "children": int(children),
        "babies": int(babies),
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": int(is_repeated_guest),
        "previous_cancellations": int(previous_cancellations),
        "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
        "reserved_room_type": reserved_room_type,
        "booking_changes": int(booking_changes),
        "deposit_type": deposit_type,
        # Missing agents were the category "Unknown" in training (NaN here)
        "agent": np.nan if agent_choice in ("Unknown", 0) else int(agent_choice),
        "customer_type": customer_type,
        "adr": float(adr),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
    }

    # Use the schema's training-time column order when it is available.
    input_columns = schema.get("input_columns", list(input_data.keys()))
    missing_columns = [column for column in input_columns if column not in input_data]

    if missing_columns:
        st.error(
            "The model schema contains input columns that are not provided "
            f"by this app: {missing_columns}"
        )
        st.stop()

    input_df = pd.DataFrame(
        [[input_data[column] for column in input_columns]],
        columns=input_columns,
    )

    try:
        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction result")

        if prediction == 1:
            st.error("⚠️ The model predicts that this booking is likely to be cancelled.")
        else:
            st.success("✅ The model predicts that this booking is likely to be retained.")

        # Models were trained with class_weight="balanced", so these scores lean
        # towards "cancel" and are a risk score, not a calibrated probability.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if 1 in classes:
                cancellation_probability = float(probabilities[classes.index(1)])
                retention_probability = 1.0 - cancellation_probability

                m1, m2 = st.columns(2)
                m1.metric("Cancellation risk score", f"{cancellation_probability:.1%}")
                m2.metric("Retention score", f"{retention_probability:.1%}")

                st.progress(
                    max(0.0, min(1.0, cancellation_probability)),
                    text="Estimated cancellation risk",
                )
        else:
            st.info(
                "This saved model does not provide calibrated probabilities; "
                "the predicted class is shown above."
            )

        with st.expander("View submitted booking details"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "This prediction is an estimate from the trained machine-learning "
            "model, not a guarantee of the booking outcome. The risk score is "
            "not a calibrated probability."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that this model file was created from the same training "
            "pipeline and that its required dependencies are installed."
        )


