import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

AVIS_COLORS = {
    "FML": "#00285F",
    "OLMM": "#D50025",
    "PL": "#FF7513"
}

@st.cache_data
def upload_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def create_stacked_bar_chart(df_data, index_col, value_col, title, prefix=""):
    pivot_df = df_data.pivot_table(
        index=index_col,
        columns="PRODUCT CODE",
        values=value_col,
        aggfunc="sum" if value_col == "CONVERTED RENTAL" else "count",
        fill_value=0
    ).reset_index()

    product_cols = [col for col in pivot_df.columns if col not in [index_col, "Grand Total"]]
    pivot_df["Grand Total"] = pivot_df[product_cols].sum(axis=1)

    fig = px.bar(
        pivot_df,
        x=index_col,
        y=product_cols,
        barmode="stack",
        title=title,
        color_discrete_map=AVIS_COLORS
    )

    fig.update_yaxes(title_text=f"{'Revenue' if value_col == 'CONVERTED RENTAL' else 'Fleet Count'}", tickprefix=prefix, tickformat=",.0f")
    
    fig.update_traces(
        texttemplate=f"{prefix}" + "%{y:,.0f}" if prefix else "%{y:,.0f}",
        textposition='inside',
        insidetextanchor='middle'
    )

    for _, row in pivot_df.iterrows():
        fig.add_annotation(
            x=row[index_col],
            y=row["Grand Total"],
            text=f"{prefix}{row['Grand Total']:,.0f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            yshift=10
        )
    
    return fig

st.markdown(
    """
    <h1 style='color: #D40029;
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 8px;'>
                AVIS FLEET DASHBOARD
    </h1>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    df = None
    if uploaded_file:
        df = upload_data(uploaded_file)
        if df is not None:
            st.success("File uploaded successfully!")
        else:
            st.error("There was an error processing the file. Please check the format.")
    
if df is not None:
    st.markdown("---")
    
    total_fleet_count = df["MVA NUMBER"].count()
    total_revenue = df["CONVERTED RENTAL"].sum()
    total_admin_fees = df["CONVERTED ADMIN FEE"].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Fleet Count", value=f"{total_fleet_count:,}")
        
    with col2:
        st.metric(label="Total Revenue", value=f"R {total_revenue:,.0f}")

    with col3:
        st.metric(label="Total Admin Fees", value=f"R {total_admin_fees:,.0f}")
        
    st.markdown("---")
    
    tabs = st.tabs(["Summary", "Revenue", "Fleet Count", "Regional Summary", "Fleet Heatmap", "Maintenance"])

    with tabs[0]:
        with st.expander("Data Preview"):
            st.dataframe(df)

        required_cols_funnel = {"COMPANY CODE", "CUSTOMER TRADING NAME", "CONVERTED RENTAL", "MVA NUMBER"}
        if required_cols_funnel.issubset(df.columns):
            
            countries = st.multiselect(
                "Filter by Country",
                options=sorted(df["COMPANY CODE"].unique()),
                default=sorted(df["COMPANY CODE"].unique())
            )
            
            filtered_df = df[df["COMPANY CODE"].isin(countries)].copy()
            
            if not filtered_df.empty:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.header("Revenue by Customer (Top 10)")
                    revenue_by_customer = filtered_df.groupby("CUSTOMER TRADING NAME")["CONVERTED RENTAL"].sum().reset_index()
                    top_10_customers_revenue = revenue_by_customer.sort_values(by="CONVERTED RENTAL", ascending=False).head(10)
                    
                    fig_funnel_revenue = px.funnel(
                        top_10_customers_revenue,
                        x="CONVERTED RENTAL",
                        y="CUSTOMER TRADING NAME",
                        title="Top 10 Customers by Revenue",
                        labels={"CONVERTED RENTAL": "Revenue", "CUSTOMER TRADING NAME": "Customer"},
                        color_discrete_sequence=["#00285F"]
                    )
                    
                    fig_funnel_revenue.update_traces(
                        textposition="inside",
                        textinfo="value",
                        texttemplate="R %{value:,.0f}"
                    )
                    
                    fig_funnel_revenue.update_layout(
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_funnel_revenue, use_container_width=True)

                with col_chart2:
                    st.header("Fleet Count by Customer (Top 10)")
                    fleet_count_by_customer = filtered_df.groupby("CUSTOMER TRADING NAME")["MVA NUMBER"].count().reset_index()
                    top_10_customers_fleet = fleet_count_by_customer.sort_values(by="MVA NUMBER", ascending=False).head(10)

                    fig_funnel_fleet = px.funnel(
                        top_10_customers_fleet,
                        x="MVA NUMBER",
                        y="CUSTOMER TRADING NAME",
                        title="Top 10 Customers by Fleet Count",
                        labels={"MVA NUMBER": "Fleet Count", "CUSTOMER TRADING NAME": "Customer"},
                        color_discrete_sequence=["#00285F"]
                    )

                    fig_funnel_fleet.update_traces(
                        textposition="inside",
                        textinfo="value",
                        texttemplate="%{value:,.0f}"
                    )

                    fig_funnel_fleet.update_layout(
                        showlegend=False
                    )

                    st.plotly_chart(fig_funnel_fleet, use_container_width=True)
            else:
                st.info("No data available for the selected country(s).")
        else:
            st.warning(f"Data must contain the following columns for this tab: {list(required_cols_funnel)}")

    with tabs[1]:
        st.header("Revenue Analysis")
        required_cols = {"COMPANY CODE", "PRODUCT CODE", "CONVERTED RENTAL"}
        if required_cols.issubset(df.columns):
            col_rev1, col_rev2 = st.columns(2)
            with col_rev1:
                greater_africa_df = df[df['COMPANY CODE'] != "RSA"].copy()
                greater_africa_df['CONVERTED RENTAL'] = pd.to_numeric(greater_africa_df['CONVERTED RENTAL'], errors='coerce')
                greater_africa_df.dropna(subset=['CONVERTED RENTAL'], inplace=True)
                
                fig1 = create_stacked_bar_chart(greater_africa_df, "COMPANY CODE", "CONVERTED RENTAL", "Greater Africa Revenue", "R ")
                st.plotly_chart(fig1, use_container_width=True)

            with col_rev2:
                rsa_df = df[df['COMPANY CODE'] == "RSA"].copy()
                rsa_df['CONVERTED RENTAL'] = pd.to_numeric(rsa_df['CONVERTED RENTAL'], errors='coerce')
                rsa_df.dropna(subset=['CONVERTED RENTAL'], inplace=True)
                
                fig2 = create_stacked_bar_chart(rsa_df, "REGION", "CONVERTED RENTAL", "RSA Revenue", "R ")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"Data must contain the following columns for this tab: {list(required_cols)}")

    with tabs[2]:
        st.header("Fleet Count Analysis")
        required_cols = {"COMPANY CODE", "PRODUCT CODE", "MVA NUMBER"}
        if required_cols.issubset(df.columns):
            col_fleet1, col_fleet2 = st.columns(2)
            with col_fleet1:
                greater_africa_df = df[df['COMPANY CODE'] != "RSA"].copy()
                
                fig1 = create_stacked_bar_chart(greater_africa_df, "COMPANY CODE", "MVA NUMBER", "Greater Africa Fleet Count")
                st.plotly_chart(fig1, use_container_width=True)

            with col_fleet2:
                rsa_df = df[df['COMPANY CODE'] == "RSA"].copy()
                
                fig2 = create_stacked_bar_chart(rsa_df, "REGION", "MVA NUMBER", "RSA Fleet Count")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"Data must contain the following columns for this tab: {list(required_cols)}")

    with tabs[3]:
        st.header("Regional Summary")
        
        # Create a copy of the dataframe to filter
        filtered_df = df.copy()

        with st.expander("Filter Options"):
            if "VEHICLE TYPE DESCRIPTION" in df.columns:
                vehicle_types = st.multiselect(
                    "Select Vehicle Type(s)",
                    options=sorted(filtered_df["VEHICLE TYPE DESCRIPTION"].unique()),
                    default=[]  # No default selection
                )
                if vehicle_types:
                    filtered_df = filtered_df[filtered_df["VEHICLE TYPE DESCRIPTION"].isin(vehicle_types)]
            else:
                vehicle_types = []
            
            if "MAKE DESCRIPTION" in df.columns:
                makes = st.multiselect(
                    "Select Make(s)",
                    options=sorted(filtered_df["MAKE DESCRIPTION"].unique()),
                    default=[] # No default selection
                )
                if makes:
                    filtered_df = filtered_df[filtered_df["MAKE DESCRIPTION"].isin(makes)]
            else:
                makes = []

            if "RANGE DESCRIPTION" in df.columns:
                ranges = st.multiselect(
                    "Select Range(s)",
                    options=sorted(filtered_df["RANGE DESCRIPTION"].unique()),
                    default=[]
                )
                if ranges:
                    filtered_df = filtered_df[filtered_df["RANGE DESCRIPTION"].isin(ranges)]
            else:
                ranges = []

            if "MODEL DESCRIPTION" in df.columns:
                models = st.multiselect(
                    "Select Model(s)",
                    options=sorted(filtered_df["MODEL DESCRIPTION"].unique()),
                    default=[] # No default selection
                )
                if models:
                    filtered_df = filtered_df[filtered_df["MODEL DESCRIPTION"].isin(models)]
            else:
                models = []
            
            if "PRODUCT CODE" in df.columns:
                product_codes = st.multiselect(
                    "Select Product Code(s)",
                    options=sorted(filtered_df["PRODUCT CODE"].unique()),
                    default=[] # No default selection
                )
                if product_codes:
                    filtered_df = filtered_df[filtered_df["PRODUCT CODE"].isin(product_codes)]
            else:
                product_codes = []
        
        # Check if the required columns exist in the filtered data before plotting
        if not filtered_df.empty and {"COMPANY CODE", "CONVERED INTEREST AMOUNT", "CONVERTED ADMIN FEE", "CONVERTED RENTAL", "MVA NUMBER"}.issubset(filtered_df.columns):
            # Group by 'COMPANY CODE' to get the summary metrics
            summary_df = filtered_df.groupby("COMPANY CODE").agg(
                **{
                    "Avg Interest": pd.NamedAgg(column="CONVERED INTEREST AMOUNT", aggfunc="mean"),
                    "Avg Admin": pd.NamedAgg(column="CONVERTED ADMIN FEE", aggfunc="mean"),
                    "Avg Rental Amount": pd.NamedAgg(column="CONVERTED RENTAL", aggfunc="mean"),
                    "Fleet Count": pd.NamedAgg(column="MVA NUMBER", aggfunc="count"),
                }
            ).reset_index()

            company_codes = summary_df["COMPANY CODE"]

            # Create a subplot with two y-axes for the mixed chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bar traces for the average financial amounts with text labels
            fig.add_trace(go.Bar(x=company_codes, y=summary_df["Avg Interest"], name="Avg Interest", marker_color='#FF7F0E', text=summary_df["Avg Interest"].apply(lambda x: f"R {x:,.0f}"), textposition='outside'), secondary_y=False)
            fig.add_trace(go.Bar(x=company_codes, y=summary_df["Avg Admin"], name="Avg Admin", marker_color='#1F3A5E', text=summary_df["Avg Admin"].apply(lambda x: f"R {x:,.0f}"), textposition='outside'), secondary_y=False)
            fig.add_trace(go.Bar(x=company_codes, y=summary_df["Avg Rental Amount"], name="Avg Rental Amount", marker_color='#A9A9A9', text=summary_df["Avg Rental Amount"].apply(lambda x: f"R {x:,.0f}"), textposition='outside'), secondary_y=False)

            # Add the line trace for the fleet count with text labels
            fig.add_trace(go.Scatter(x=company_codes, y=summary_df["Fleet Count"], name="Fleet Count", marker=dict(color='#D62728'), line=dict(width=3), text=summary_df["Fleet Count"], textposition='top center', mode='lines+markers+text'), secondary_y=True)

            # Update the chart layout and titles
            fig.update_layout(
                title_text="Regional Summary",
                xaxis_title="COMPANY CODE",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Set the y-axis titles for both axes
            fig.update_yaxes(title_text="<b>Amount</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Fleet Count</b>", secondary_y=True)

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please upload data and ensure it contains 'COMPANY CODE', 'CONVERED INTEREST AMOUNT', 'CONVERTED ADMIN FEE', 'CONVERTED RENTAL', and 'MVA NUMBER' columns for this chart.")
            
    with tabs[4]:
        st.header("Contract Parameters")
        
        required_cols_heatmap = {"COMPANY CODE", "LAST CONTRACT MONTHS", "LAST CONTRACT KM", "MVA NUMBER", "VEHICLE TYPE DESCRIPTION"}
        
        if required_cols_heatmap.issubset(df.columns):
            company_code_filter = st.selectbox(
                "Filter by Company Code",
                options=sorted(df["COMPANY CODE"].unique())
            )

            filtered_df_heatmap = df[df["COMPANY CODE"] == company_code_filter].copy()

            vehicle_types = st.multiselect(
                "Filter by Vehicle Type",
                options=sorted(filtered_df_heatmap["VEHICLE TYPE DESCRIPTION"].unique())
            )
            
            if vehicle_types:
                filtered_df_heatmap = filtered_df_heatmap[filtered_df_heatmap["VEHICLE TYPE DESCRIPTION"].isin(vehicle_types)]

            if not filtered_df_heatmap.empty:
                
                month_bins = [0, 12, 24, 36, 48, 60, 72, 84, float('inf')]
                month_labels = ["<12", "12-24", "24-36", "36-48", "48-60", "60-72", "72-84", ">84"]
                
                km_bins = [0, 50000, 100000, 150000, 200000, 250000, 300000, float('inf')]
                km_labels = ["<50k", "50k-100k", "100k-150k", "150k-200k", "200k-250k", "250k-300k", ">300k"]

                filtered_df_heatmap["Months_Bins"] = pd.cut(filtered_df_heatmap["LAST CONTRACT MONTHS"], bins=month_bins, labels=month_labels, right=False)
                filtered_df_heatmap["KMs_Bins"] = pd.cut(filtered_df_heatmap["LAST CONTRACT KM"], bins=km_bins, labels=km_labels, right=False)

                heatmap_data = filtered_df_heatmap.pivot_table(
                    index="KMs_Bins",
                    columns="Months_Bins",
                    values="MVA NUMBER",
                    aggfunc="count",
                    fill_value=0
                ).reindex(index=km_labels, columns=month_labels)

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Blues'
                ))
                
                for i in range(len(heatmap_data.index)):
                    for j in range(len(heatmap_data.columns)):
                        value = heatmap_data.iloc[i, j]
                        fig_heatmap.add_annotation(
                            x=heatmap_data.columns[j],
                            y=heatmap_data.index[i],
                            text=str(int(value)),
                            showarrow=False,
                            font=dict(color='black' if value < heatmap_data.max().max() * 0.5 else 'white')
                        )

                fig_heatmap.update_layout(
                    title="Fleet Distribution Heatmap",
                    xaxis_title="Contract Months",
                    yaxis_title="Contract Kilometers",
                )

                avg_months = filtered_df_heatmap["LAST CONTRACT MONTHS"].mean()
                avg_km = filtered_df_heatmap["LAST CONTRACT KM"].mean()
                
                col_avg_m, col_avg_k = st.columns(2)
                with col_avg_m:
                    st.metric(label="Average Months", value=f"{avg_months:,.0f}")
                with col_avg_k:
                    st.metric(label="Average KMS", value=f"{avg_km:,.0f}")
                
                st.plotly_chart(fig_heatmap, use_container_width=True)

            else:
                st.info("No data available for the selected filters.")
        else:
            st.warning(f"Data must contain the following columns for this tab: {list(required_cols_heatmap)}")
            
    with tabs[5]:
        st.header("Maintenance Analysis")
    
        required_cols_maintenance = {"COMPANY CODE", "CONVERTED MAINTENANCE2"}
    
        if required_cols_maintenance.issubset(df.columns):
            total_positive_profit = df['CONVERTED MAINTENANCE2'].loc[df['CONVERTED MAINTENANCE2'] > 0].sum()
            total_negative_profit = df['CONVERTED MAINTENANCE2'].loc[df['CONVERTED MAINTENANCE2'] < 0].sum()
            total_maintenance = df['CONVERTED MAINTENANCE2'].sum()
    
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label="Total Profitable Maintenance Fund", value=f"R {total_positive_profit:,.0f}")
            with col_m2:
                st.metric(label="Total Buning Maintenance Fund", value=f"R {total_negative_profit:,.0f}")
            with col_m3:
                st.metric(label="Total Maintenance Fund Balance", value=f"R {total_maintenance:,.0f}")
    
            st.markdown("---")
            
            col1, col2 = st.columns(2)
    
            with col1:
                st.subheader("Greater Africa")
                greater_africa_df = df[df['COMPANY CODE'] != "RSA"].copy()
                
                if not greater_africa_df.empty:
                    greater_africa_df['Profit Status'] = greater_africa_df['CONVERTED MAINTENANCE2'].apply(
                        lambda x: 'Profitable' if x > 0 else 'Burning'
                    )
                    maintenance_summary_ga = greater_africa_df.groupby(['COMPANY CODE', 'Profit Status'])['CONVERTED MAINTENANCE2'].sum().reset_index(name='Total Profit')
                    
                    fig_maintenance_ga = px.bar(
                        maintenance_summary_ga,
                        x="COMPANY CODE",
                        y="Total Profit",
                        color="Profit Status",
                        title="Profitable vs. Burning Maintenance Fund",
                        color_discrete_map={'Profitable': '#71C562', 'Burning': '#F32013'}
                    )
                    fig_maintenance_ga.update_traces(
                        texttemplate='R%{y:,.0f}',
                        textposition='inside'
                    )
                    fig_maintenance_ga.update_yaxes(title_text="Total Profit")
                    st.plotly_chart(fig_maintenance_ga, use_container_width=True)
                else:
                    st.info("No data available for Greater Africa.")
    
            with col2:
                st.subheader("RSA")
                rsa_df = df[df['COMPANY CODE'] == "RSA"].copy()
    
                if not rsa_df.empty:
                    rsa_df['Profit Status'] = rsa_df['CONVERTED MAINTENANCE2'].apply(
                        lambda x: 'Profitable' if x > 0 else 'Burning'
                    )
                    maintenance_summary_rsa = rsa_df.groupby(['COMPANY CODE', 'Profit Status'])['CONVERTED MAINTENANCE2'].sum().reset_index(name='Total Profit')
    
                    fig_maintenance_rsa = px.bar(
                        maintenance_summary_rsa,
                        x="COMPANY CODE",
                        y="Total Profit",
                        color="Profit Status",
                        title="Profitable vs. Burning Maintenance Fund (RSA)",
                        color_discrete_map={'Profitable': '#71C562', 'Burning': '#F32013'}
                    )
                    fig_maintenance_rsa.update_traces(
                        texttemplate='R%{y:,.0f}',
                        textposition='inside'
                    )
                    fig_maintenance_rsa.update_yaxes(title_text="Total Profit")
                    st.plotly_chart(fig_maintenance_rsa, use_container_width=True)
                else:
                    st.info("No data available for RSA.")
                
        else:
            st.warning(f"Data must contain the following columns for this tab: {list(required_cols_maintenance)}")
                

else:
    st.info("Please upload an Excel file to see the dashboard.")