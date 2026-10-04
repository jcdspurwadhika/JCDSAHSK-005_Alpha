import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Alpha Hotel Revenue Protection Dashboard",
    page_icon="🏨",
    layout="wide"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

/* =====================================================
APP BACKGROUND
===================================================== */

.stApp {
    background-color: #f3f5f7;
}

/* =====================================================
HEADER
===================================================== */

h1 {
    color: #2F3B52;
    font-weight: 700;
}

h2, h3, h4 {
    color: #2F3B52;
}

/* =====================================================
KPI CARDS
===================================================== */

div[data-testid="metric-container"] {

    background-color: white;

    padding: 18px;

    border-radius: 12px;

    border: 1px solid #D8DDE6;

    box-shadow:
    0px 2px 8px rgba(
        0,
        0,
        0,
        0.05
    );
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background-color: #5F6E86;
}

/* =====================================================
SIDEBAR HEADINGS
===================================================== */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {

    color: white !important;
}

/* =====================================================
SIDEBAR LABELS
===================================================== */

section[data-testid="stSidebar"] label {

    color: #F8FAFC !important;

    font-weight: 600;
}

/* =====================================================
SIDEBAR TEXT
===================================================== */

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {

    color: #DCE3EC !important;

    font-size: 14px;

    line-height: 1.7;
}

/* =====================================================
SELECT BOXES
===================================================== */

section[data-testid="stSidebar"] div[data-baseweb="select"] {

    background: rgba(
        255,
        255,
        255,
        0.15
    ) !important;

    border-radius: 10px !important;

    border: 1px solid rgba(
        255,
        255,
        255,
        0.2
    ) !important;
}

/* Dropdown Text */

section[data-testid="stSidebar"] div[data-baseweb="select"] * {

    color: #1F2937 !important;

    font-weight: 500;
}

/* =====================================================
NUMBER INPUTS
===================================================== */

section[data-testid="stSidebar"] input {

    background: rgba(
        255,
        255,
        255,
        0.15
    ) !important;

    color: #1F2937 !important;

    border-radius: 10px !important;

    border: 1px solid rgba(
        255,
        255,
        255,
        0.2
    ) !important;
}

/* =====================================================
BUTTONS
===================================================== */

section[data-testid="stSidebar"] button {

    border-radius: 10px;

    font-weight: 600;
}

/* =====================================================
TABS
===================================================== */

button[data-baseweb="tab"] {

    font-size: 14px;

    font-weight: 600;
}

/* =====================================================
INFO / ALERT BOX
===================================================== */

[data-testid="stAlert"] {

    border-radius: 12px;
}

/* =====================================================
DATAFRAME
===================================================== */

[data-testid="stDataFrame"] {

    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    with open(
        "soft_voting_threshold_048.pkl",
        "rb"
    ) as file:

        deployment_bundle = pickle.load(file)

    return (
        deployment_bundle["model"],
        deployment_bundle["threshold"]
    )

model, threshold = load_model()



# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "hotel_bookings.csv"
    )

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "🏨 Alpha Hotel"
)

st.sidebar.markdown("""
### Project Information

**Model**
- Soft Voting Ensemble

**Business Objective**
- Revenue Protection
- Revenue At Risk

**Developer**
- Rosalina
- Aaron Avniel Siwi
- Alifian Diaz Islamy
""")

st.sidebar.markdown("---")

st.sidebar.header(
    "Booking Information"
)

hotel = st.sidebar.selectbox(
    "Hotel",
    [
        "City Hotel",
        "Resort Hotel"
    ]
)

market_segment = st.sidebar.selectbox(
    "Market Segment",
    [
        "Online TA",
        "Offline TA/TO",
        "Direct",
        "Corporate",
        "Groups",
        "Aviation",
        "Complementary",
        "Undefined"
    ]
)

distribution_channel = st.sidebar.selectbox(
    "Distribution Channel",
    [
        "TA/TO",
        "Direct",
        "Corporate",
        "GDS",
        "Undefined"
    ]
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    [
        "No Deposit",
        "Refundable",
        "Non Refund"
    ]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    [
        "Transient",
        "Transient-Party",
        "Contract",
        "Group"
    ]
)

st.sidebar.markdown("---")

st.sidebar.header(
    "Guest Information"
)

adults = st.sidebar.number_input(
    "Adults",
    min_value=0,
    value=2
)

children = st.sidebar.number_input(
    "Children",
    min_value=0,
    value=0
)

babies = st.sidebar.number_input(
    "Babies",
    min_value=0,
    value=0
)

st.sidebar.markdown("---")

st.sidebar.header(
    "Reservation Information"
)

lead_time = st.sidebar.number_input(
    "Lead Time",
    min_value=0,
    value=50
)

week_nights = st.sidebar.number_input(
    "Week Nights",
    min_value=0,
    value=3
)

weekend_nights = st.sidebar.number_input(
    "Weekend Nights",
    min_value=0,
    value=1
)

previous_cancellations = st.sidebar.number_input(
    "Previous Cancellations",
    min_value=0,
    value=0
)

special_requests = st.sidebar.number_input(
    "Special Requests",
    min_value=0,
    value=1
)

st.sidebar.markdown("---")

st.sidebar.header(
    "Revenue Information"
)

adr = st.sidebar.number_input(
    "ADR",
    min_value=0.0,
    value=100.0
)

predict_btn = st.sidebar.button(
    "🔮 Predict Booking Risk",
    use_container_width=True
)

# =====================================================
# HEADER
# =====================================================

st.title(
    "🏨 Alpha Hotel Revenue Protection Dashboard"
)

st.markdown("""
Early Warning System for High-Risk Booking Cancellations

This dashboard combines cancellation prediction,
Revenue At Risk estimation,
and business-oriented intervention recommendations.
""")

# =====================================================
# KPI CARDS
# =====================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Recall",
        "87.85%"
    )

with k2:
    st.metric(
        "F2 Score",
        "80.68%"
    )

with k3:
    st.metric(
        "ROC AUC",
        "91.34%"
    )

with k4:
    st.metric(
        "Threshold",
        "0.48"
    )

st.markdown("---")

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔮 Risk Assessment",
        "💰 Revenue At Risk",
        "📊 Historical Analytics",
        "🤖 Model Insights",
        "📋 Recommendation Center"
    ]
)

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader(
        "Booking Risk Assessment"
    )

    if predict_btn:

        # ==========================================
        # FEATURE ENGINEERING
        # ==========================================

        booking_revenue = (
            adr *
            (
                week_nights +
                weekend_nights
            )
        )

        total_nights = (
            week_nights +
            weekend_nights
        )

        total_guests = (
            adults +
            children +
            babies
        )

        net_cancelled = (
            previous_cancellations -
            0
        )

        rev_per_guest = (
            booking_revenue /
            max(total_guests, 1)
        )

        high_lead_time = (
            1 if lead_time > 100 else 0
        )

        # ==========================================
        # MODEL INPUT
        # ==========================================

        input_df = pd.DataFrame({

            "hotel":[hotel],

            "lead_time":[lead_time],

            "arrival_date_year":[2017],

            "arrival_date_month":["July"],

            "arrival_date_week_number":[30],

            "arrival_date_day_of_month":[15],

            "stays_in_weekend_nights":[weekend_nights],

            "stays_in_week_nights":[week_nights],

            "adults":[adults],

            "children":[children],

            "babies":[babies],

            "meal":["BB"],

            "country":["PRT"],

            "market_segment":[market_segment],

            "distribution_channel":[distribution_channel],

            "is_repeated_guest":[0],

            "previous_cancellations":[
                previous_cancellations
            ],

            "previous_bookings_not_canceled":[0],

            "reserved_room_type":["A"],

            "assigned_room_type":["A"],

            "booking_changes":[0],

            "deposit_type":[deposit_type],

            "agent":["9.0"],

            "days_in_waiting_list":[0],

            "customer_type":[customer_type],

            "adr":[adr],

            "required_car_parking_spaces":[0],

            "total_of_special_requests":[
                special_requests
            ],

            "is_company_booking":[0],

            "total_nights":[total_nights],

            "total_guests":[total_guests],

            "booking_revenue":[booking_revenue],

            "net_cancelled":[net_cancelled],

            "rev_per_guest":[rev_per_guest],

            "high_lead_time":[high_lead_time]
        })

        # ==========================================
        # REAL PREDICTION
        # ==========================================

        try:

            cancel_prob = (
                model.predict_proba(
                    input_df
                )[0][1]
            )

        except Exception as e:

            st.error(
                f"Prediction Error : {e}"
            )

            st.stop()

        # ==========================================
        # RISK LEVEL
        # ==========================================

        if cancel_prob >= 0.80:

            risk_level = "HIGH RISK"

        elif cancel_prob >= threshold:

            risk_level = "MEDIUM RISK"

        else:

            risk_level = "LOW RISK"

        # ==========================================
        # KPI
        # ==========================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Cancellation Probability",
                f"{cancel_prob:.2%}"
            )

        with c2:

            st.metric(
                "Production Threshold",
                f"{threshold:.2f}"
            )

        with c3:

            st.metric(
                "Risk Level",
                risk_level
            )

        # ==========================================
        # GAUGE
        # ==========================================

        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=cancel_prob * 100,
                gauge={
                    "axis":{
                        "range":[0,100]
                    },

                    "bar":{
                        "color":"#56637A"
                    },

                    "steps":[
                        {
                            "range":[0,48],
                            "color":"#DCE3EC"
                        },
                        {
                            "range":[48,80],
                            "color":"#AAB7CA"
                        },
                        {
                            "range":[80,100],
                            "color":"#72809A"
                        }
                    ],

                    "threshold":{
                        "line":{
                            "color":"black",
                            "width":4
                        },
                        "value":48
                    }
                }
            )
        )

        fig.update_layout(
            height=180,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            )
        )

        l, c, r = st.columns(
            [2,3,2]
        )

        with c:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ==========================================
        # ALERT
        # ==========================================

        if cancel_prob >= 0.80:

            st.error(
                "🚨 HIGH RISK BOOKING"
            )

        elif cancel_prob >= threshold:

            st.warning(
                "⚠ MEDIUM RISK BOOKING"
            )

        else:

            st.success(
                "✅ LOW RISK BOOKING"
            )

    else:

        st.info(
            "Complete the booking information and press Predict Booking Risk."
        )

        
# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader(
        "Revenue At Risk Analysis"
    )

    if predict_btn:

        revenue_at_risk = (
            booking_revenue *
            cancel_prob
        )

        exposure_pct = (
            revenue_at_risk /
            booking_revenue
        ) * 100

        remaining_revenue = (
            booking_revenue -
            revenue_at_risk
        )

        # ==========================================
        # KPI CARDS
        # ==========================================

        k1, k2, k3 = st.columns(3)

        with k1:

            st.metric(
                "Booking Revenue",
                f"${booking_revenue:,.2f}"
            )

        with k2:

            st.metric(
                "Cancellation Probability",
                f"{cancel_prob:.2%}"
            )

        with k3:

            st.metric(
                "Revenue At Risk",
                f"${revenue_at_risk:,.2f}"
            )

        st.markdown("---")

        # ==========================================
        # DONUT + KPI INSIGHT
        # ==========================================

        left_col, right_col = st.columns(
            [1.4, 0.6]
        )

        with left_col:

            donut_df = pd.DataFrame({
                "Category": [
                    "Revenue At Risk",
                    "Remaining Revenue"
                ],
                "Value": [
                    revenue_at_risk,
                    remaining_revenue
                ]
            })

            fig = px.pie(
                donut_df,
                names="Category",
                values="Value",
                hole=0.65,
                color="Category",
                color_discrete_map={
                    "Revenue At Risk": "#7B93B6",
                    "Remaining Revenue": "#DCE3EC"
                }
            )

            fig.update_layout(
                title={
                    "text": "Revenue Exposure Composition",
                    "x": 0.5
                },
                height=420,
                legend=dict(
                    orientation="h",
                    y=-0.15,
                    x=0.5,
                    xanchor="center"
                ),
                margin=dict(
                    l=0,
                    r=0,
                    t=50,
                    b=60
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right_col:

            st.metric(
                "Exposure Ratio",
                f"{exposure_pct:.1f}%"
            )

            st.metric(
                "Revenue At Risk",
                f"${revenue_at_risk:,.0f}"
            )

            st.metric(
                "Remaining Revenue",
                f"${remaining_revenue:,.0f}"
            )

            st.markdown("---")

            if exposure_pct >= 75:

                st.error(
                    "🚨 Critical Exposure"
                )

            elif exposure_pct >= 50:

                st.warning(
                    "⚠ Moderate Exposure"
                )

            else:

                st.success(
                    "✅ Low Exposure"
                )

        st.markdown("---")

        # ==========================================
        # RECOMMENDATION
        # ==========================================

        if exposure_pct >= 75:

            st.error(
                f"""
### Revenue Exposure Alert

Potential revenue exposure:

**${revenue_at_risk:,.2f}**

Recommended Actions:

✅ Contact customer immediately

✅ Send confirmation request

✅ Prioritize retention campaign

✅ Revenue protection required
"""
            )

        elif exposure_pct >= 50:

            st.warning(
                f"""
### Moderate Revenue Exposure

Potential revenue exposure:

**${revenue_at_risk:,.2f}**

Recommended Actions:

✅ Customer follow-up

✅ Confirmation reminder

✅ Booking verification
"""
            )

        else:

            st.success(
                f"""
### Low Revenue Exposure

Potential revenue exposure:

**${revenue_at_risk:,.2f}**

Recommended Actions:

✅ Standard monitoring is sufficient
"""
            )

    else:

        st.info(
            "Generate prediction first."
        )
# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.subheader(
        "Historical Analytics"
    )

    # ==========================================
    # KPI
    # ==========================================

    k1, k2, k3 = st.columns(3)

    with k1:

        st.metric(
            "Total Bookings",
            f"{len(df):,}"
        )

    with k2:

        cancel_rate = (
            df["is_canceled"]
            .mean()
            * 100
        )

        st.metric(
            "Cancellation Rate",
            f"{cancel_rate:.2f}%"
        )

    with k3:

        avg_adr = (
            df["adr"]
            .mean()
        )

        st.metric(
            "Average ADR",
            f"${avg_adr:.2f}"
        )

    st.markdown("---")

    # ==========================================
    # CORPORATE COLORS
    # ==========================================

    PRIMARY_BLUE = "#6E86B3"
    SECONDARY_BLUE = "#90A8D0"
    LIGHT_BLUE = "#DCE5F2"
    DARK_BLUE = "#56637A"

    # ==========================================
    # ROW 1
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        hotel_cancel = (
            df.groupby("hotel")
            ["is_canceled"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig_hotel = px.bar(
            hotel_cancel,
            x="hotel",
            y="is_canceled",
            text="is_canceled",
            title="Cancellation Rate by Hotel",
            color_discrete_sequence=[
                PRIMARY_BLUE
            ]
        )

        fig_hotel.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside"
        )

        fig_hotel.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig_hotel,
            use_container_width=True
        )

    with col2:

        deposit_cancel = (
            df.groupby("deposit_type")
            ["is_canceled"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig_dep = px.bar(
            deposit_cancel,
            x="deposit_type",
            y="is_canceled",
            text="is_canceled",
            title="Cancellation Rate by Deposit Type",
            color_discrete_sequence=[
                SECONDARY_BLUE
            ]
        )

        fig_dep.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside"
        )

        fig_dep.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig_dep,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # ROW 2
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        segment_df = (
            df["market_segment"]
            .value_counts()
            .reset_index()
        )

        segment_df.columns = [
            "Market Segment",
            "Count"
        ]

        fig_segment = px.pie(
            segment_df,
            names="Market Segment",
            values="Count",
            hole=0.65,
            title="Market Segment Distribution",
            color_discrete_sequence=[
                PRIMARY_BLUE,
                SECONDARY_BLUE,
                "#AFC2D9",
                "#DCE5F2",
                "#8497B5",
                "#AEBBD2",
                "#7389AE",
                "#D2DAE8"
            ]
        )

        fig_segment.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350
        )

        st.plotly_chart(
            fig_segment,
            use_container_width=True
        )

    with col2:

        channel_df = (
            df["distribution_channel"]
            .value_counts()
            .reset_index()
        )

        channel_df.columns = [
            "Distribution Channel",
            "Count"
        ]

        fig_channel = px.bar(
            channel_df,
            x="Distribution Channel",
            y="Count",
            text="Count",
            title="Booking Distribution Channel",
            color_discrete_sequence=[
                DARK_BLUE
            ]
        )

        fig_channel.update_traces(
            textposition="outside"
        )

        fig_channel.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=420,
            showlegend=False
        )

        st.plotly_chart(
            fig_channel,
            use_container_width=True
        )

    st.markdown("---")

    st.success(
        """
### Key Business Insights

✅ City Hotels exhibit higher cancellation rates than Resort Hotels.

✅ Deposit policy is strongly associated with cancellation behavior.

✅ Online TA contributes a significant portion of bookings.

✅ Distribution channels influence booking stability and should be considered in intervention strategies.

✅ Historical trends support proactive cancellation management and Revenue At Risk prioritization.
"""
    )


# =====================================================
# TAB 4
# =====================================================

with tab4:

    st.subheader(
        "Model Performance & Insights"
    )

    # ==========================================
    # KPI CARDS
    # ==========================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Final Model",
            "Soft Voting"
        )

    with k2:

        st.metric(
            "Recall",
            "87.85%"
        )

    with k3:

        st.metric(
            "F2 Score",
            "80.68%"
        )

    with k4:

        st.metric(
            "ROC AUC",
            "91.34%"
        )

    st.markdown("---")

    # ==========================================
    # PERFORMANCE COMPARISON
    # ==========================================

    performance_df = pd.DataFrame({
        "Model": [
            "Soft Voting",
            "XGBoost",
            "LightGBM"
        ],

        "Recall": [
            0.8785,
            0.8772,
            0.8720
        ],

        "F2 Score": [
            0.8068,
            0.8057,
            0.8022
        ],

        "ROC AUC": [
            0.9134,
            0.9129,
            0.9128
        ]
    })

    comparison_chart = (
        performance_df
        .melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )
    )

    fig_performance = px.bar(
        comparison_chart,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Final Model Comparison",
        color_discrete_sequence=[
            "#6E86B3",
            "#90A8D0",
            "#DCE5F2"
        ]
    )

    fig_performance.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # TOP DRIVERS
    # ==========================================

    left_col, right_col = st.columns(
        [1,1]
    )

    with left_col:

        importance_df = pd.DataFrame({

            "Feature":[
                "Country",
                "Agent",
                "Lead Time",
                "Parking Spaces",
                "Special Requests"
            ],

            "Importance":[
                0.85,
                0.75,
                0.48,
                0.62,
                0.43
            ]
        })

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[
                "#6E86B3"
            ],
            title="Top Prediction Drivers"
        )

        fig_importance.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    with right_col:

        st.markdown(
            "### Business Interpretation"
        )

        st.info(
            """
The final Soft Voting Ensemble was selected because it achieved the strongest overall balance between:

✅ Recall

✅ F2 Score

✅ ROC AUC

The model prioritizes cancellation detection while maintaining acceptable precision and operational feasibility.
"""
        )

        st.success(
            """
Top Drivers Identified

• Country

• Agent

• Lead Time

• Required Parking Spaces

• Special Requests
"""
        )

        st.warning(
            """
Management Focus Areas

• Long Lead-Time Reservations

• Customers with Cancellation History

• High Revenue Exposure Bookings

• Online TA Reservations
"""
        )

# =====================================================
# TAB 5
# =====================================================

with tab5:

    st.subheader(
        "Executive Recommendation Center"
    )

    if predict_btn:

        st.markdown(
            """
### Revenue Protection Action Plan
"""
        )

        if cancel_prob >= 0.80:

            st.error(
                """
🚨 High Risk Booking Detected

This booking presents a significant cancellation risk
and should be prioritized immediately.
"""
            )

            r1, r2 = st.columns(2)

            with r1:

                st.success(
                    """
Recommended Actions

✅ Contact customer immediately

✅ Verify travel plans

✅ Send booking confirmation

✅ Assign priority monitoring
"""
                )

            with r2:

                st.warning(
                    f"""
Financial Impact

Estimated Revenue At Risk

**${revenue_at_risk:,.2f}**

Exposure Ratio

**{exposure_pct:.1f}%**
"""
                )

        elif cancel_prob >= 0.48:

            st.warning(
                """
⚠ Moderate Risk Booking Detected

The booking demonstrates moderate cancellation risk
and should be monitored proactively.
"""
            )

            r1, r2 = st.columns(2)

            with r1:

                st.info(
                    """
Recommended Actions

✅ Send reminder email

✅ Reconfirm reservation

✅ Monitor customer activity

✅ Schedule follow-up
"""
                )

            with r2:

                st.warning(
                    f"""
Financial Impact

Estimated Revenue At Risk

**${revenue_at_risk:,.2f}**

Exposure Ratio

**{exposure_pct:.1f}%**
"""
                )

        else:

            st.success(
                """
✅ Low Risk Booking

The booking presents limited cancellation risk.
"""
            )

            r1, r2 = st.columns(2)

            with r1:

                st.success(
                    """
Recommended Actions

✅ Normal monitoring

✅ No intervention required

✅ Standard communication process
"""
                )

            with r2:

                st.info(
                    f"""
Financial Impact

Estimated Revenue At Risk

**${revenue_at_risk:,.2f}**

Exposure Ratio

**{exposure_pct:.1f}%**
"""
                )

        st.markdown("---")

        recommendation_df = pd.DataFrame({

            "Priority":[
                "Lead Time Review",
                "Cancellation History Review",
                "Revenue At Risk Monitoring",
                "Booking Confirmation"
            ],

            "Action":[
                "Monitor long-horizon reservations",
                "Identify repeat cancellation behaviour",
                "Prioritize financially exposed reservations",
                "Reduce booking uncertainty"
            ]

        })

        st.markdown(
            "### Strategic Monitoring Checklist"
        )

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Generate prediction first to receive business recommendations."
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    """
Alpha Hotel Revenue Protection Dashboard

Powered by Soft Voting Ensemble
"""
)