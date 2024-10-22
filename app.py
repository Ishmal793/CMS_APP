import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Function to load default data
@st.cache_data
def load_default_data():
    return pd.read_excel(
        'Modified_PPE_compliance_dataset.xlsx',
        sheet_name='Sheet1',
        engine='openpyxl'
    )

# Function to load uploaded files (supports Excel and CSV)
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

# Sidebar for file upload or default dataset
st.sidebar.title("Upload or Load Dataset")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

# Load dataset based on user input
if data_source == "Default Dataset":
    data = load_default_data()
    st.sidebar.success("Default dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a dataset to proceed.")
        st.stop()

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date']).dt.date

# Add Radio Button for View Type **Above Filters**
view_type = st.sidebar.radio(
    "Select View:",
    ("Overall","Combine Unit", "Unit", "Red Zone", "Monthly Rates and Prediction", "Target")
)

# Sidebar Filters
st.sidebar.header("Filters")
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Compliance", "Violation"])

min_date, max_date = min(data['Date']), max(data['Date'])
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start Date cannot be after End Date")

employee = st.sidebar.multiselect('Select Employee', options=data['Employee_Name'].unique())
shift = st.sidebar.multiselect('Select Shift', options=data['Shift'].unique())
factory = st.sidebar.multiselect('Select Factory', options=data['Factory'].unique())
department = st.sidebar.multiselect('Select Department', options=data['Department'].unique())
camera = st.sidebar.multiselect('Select Camera', options=data['Camera'].unique())



# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_rerun()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)

# Data Filtering
filtered_data = data[
    (data['Date'] >= start_date) & (data['Date'] <= end_date) &
    (data['Employee_Name'].isin(employee) if employee else True) &
    (data['Shift'].isin(shift) if shift else True) &
    (data['Factory'].isin(factory) if factory else True) &
    (data['Department'].isin(department) if department else True) &
    (data['Camera'].isin(camera) if camera else True)
]

if analysis_type == "Violation":
    relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
    current_rate = (relevant_data.shape[0] / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Violation Rate"
    relevant_checks = relevant_data.shape[0]
else:
    relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
    compliant_checks = relevant_data.shape[0]
    current_rate = (compliant_checks / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Compliance Rate"
    relevant_checks = compliant_checks

# Prediction logic
if relevant_data.shape[0] > 0:
    relevant_data['Month'] = pd.to_datetime(relevant_data['Date']).dt.to_period('M').astype(str)
    monthly_rate = relevant_data.groupby('Month')['Violation_Type'].apply(
        lambda x: (x != 'Compliant').sum() / len(x) * 100 if analysis_type == "Violation" else
        (x == 'Compliant').sum() / len(x) * 100
    ).reset_index(name='Rate')

    try:
        coeffs = np.polyfit(range(len(monthly_rate)), monthly_rate['Rate'], 1)
        next_month_prediction = coeffs[0] * (len(monthly_rate) + 1) + coeffs[1]
    except np.linalg.LinAlgError:
        next_month_prediction = current_rate
else:
    next_month_prediction = current_rate


# Display Appropriate Heading and Charts Based on View Selection
if view_type == "Overall":
    # Select relevant data based on the analysis type
    total_checks = filtered_data.shape[0]

    # Display Header and Metrics
    st.header(f"Overall {analysis_type} Dashboard")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)


    # Display Current Rate, Next Month Prediction, Total Checks, and Relevant Checks
    col1.metric(rate_label, f"{current_rate:.2f}%")
    col2.metric("Next Month Prediction", f"{next_month_prediction:.2f}%")
    col3.metric("Total Checks", total_checks)
    col4.metric("Relevant Checks", relevant_checks)
    # Group data for visualizations
    if analysis_type == "Violation":
        grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
            Total_Violations=('Violation_Type', 'count')
        ).reset_index()
    else:
        grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
            Total_Compliance=('Violation_Type', 'count')
        ).reset_index()

    # Factory-wise Violations/Compliance Gauge
    st.subheader(f"{analysis_type} by Factory")

    factory_colors = ['#00FF00', '#FF4500', '#1E90FF', '#FFFF00',
                      '#FF1493']  # Green, OrangeRed, DodgerBlue, Yellow, DeepPink (avoiding pink now)

    # Factory-wise Violations/Compliance Gauge
    col1, col2, col3 = st.columns(3)
    for i, (factory, count) in enumerate(grouped_data.groupby('Factory')[
                                             f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum().items(),
                                         1):
        with [col1, col2, col3][i % 3]:
            color_index = i % len(factory_colors)  # Cycle through the color palette
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=count,
                title={"text": f"Factory {factory} {analysis_type}"},
                gauge={
                    'axis': {'range': [0, max(grouped_data.groupby('Factory')[
                                                  f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                    'bar': {'color': factory_colors[color_index]}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

    # Shift-wise Violations/Compliance Gauge
    st.subheader(f"{analysis_type} by Shift")

    col4, col5 = st.columns(2)
    with col4:
        shift_value = grouped_data[grouped_data['Shift'] == 'Morning'][
            f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
        fig_morning = go.Figure(go.Indicator(
            mode="gauge+number",
            value=shift_value,
            title={"text": "Morning Shift"},
            gauge={
                'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                              f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                'bar': {'color': '#32CD32'}  # LimeGreen color for Morning Shift
            }
        ))

        st.plotly_chart(fig_morning, use_container_width=True)

    with col5:
        shift_value = grouped_data[grouped_data['Shift'] == 'Evening'][
            f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
        fig_evening = go.Figure(go.Indicator(
            mode="gauge+number",
            value=shift_value,
            title={"text": "Evening Shift"},
            gauge={
                'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                              f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                'bar': {'color': '#FF8C00'}  # DarkOrange for Evening Shift
            }
        ))

        st.plotly_chart(fig_evening, use_container_width=True)
    row_selection = st.radio("Choose Rows to Display:", ("First Five Rows", "Last Five Rows"))

    # Display data based on radio selection
    if row_selection == "First Five Rows":
        st.write("### First Five Rows of the Dataset")
        st.write(data.head())
    else:
        st.write("### Last Five Rows of the Dataset")
        st.write(data.tail())
    fig = px.histogram(filtered_data, x='Date', color='Department', title=f"Overall {analysis_type} Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif view_type == "Combine Unit":
    st.subheader(f"{analysis_type} by Factory, Department")

    # Group data for Unit-wise Violations/Compliance
    grouped_unit_data = relevant_data.groupby(['Factory', 'Department']).agg(
        Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
    ).reset_index()

    # Color palette for the bar chart
    color_palette = px.colors.qualitative.Set3  # Using a vibrant color palette

    # Create a bar chart for Unit-wise Violations/Compliance
    fig_unit = px.bar(grouped_unit_data,
                      x='Factory',
                      y='Total_Count',
                      color='Department',
                      title=f"{analysis_type} by Unit",
                      labels={
                          'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                      color_discrete_sequence=color_palette)  # Assign color palette

    # Update layout for better appearance
    fig_unit.update_layout(
        xaxis_title="Factory",
        yaxis_title="Count",
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        font=dict(color="white")  # White font color for contrast with dark background
    )

    st.plotly_chart(fig_unit, use_container_width=True)

    # Filter data based on user input
    if analysis_type == "Violation":
        relevant_data = data[data['Violation_Type'] != 'Compliant']
    else:
        relevant_data = data[data['Violation_Type'] == 'Compliant']


    # Function to create combined charts
    def combined_charts():
        fig = go.Figure()

        # Group data by Department and Shift
        department_shift_data = relevant_data.groupby(['Department', 'Shift']).agg(
            Total_Count=('Violation_Type', 'count')
        ).reset_index()

        # Add Department by Shift Bar Chart
        # Using distinct colors for each department
        department_colors = px.colors.qualitative.Pastel  # Color palette for department-wise chart

        for i, department in enumerate(department_shift_data['Department'].unique()):
            department_data = department_shift_data[department_shift_data['Department'] == department]
            color = department_colors[i % len(department_colors)]  # Cycle through the color palette
            fig.add_trace(go.Bar(
                x=department_data['Shift'],
                y=department_data['Total_Count'],
                name=str(department),  # Ensure department name is used as legend entry
                hoverinfo='text',
                text=department_data['Total_Count'],
                marker_color=color  # Set the color for each department
            ))

        # Update the layout with title and axis labels
        fig.update_layout(
            title=f"{analysis_type} by Department and Shift",
            barmode='stack',  # Stacked bar chart
            xaxis_title='Shift',
            yaxis_title='Total Count',
            legend_title='Department',
            template='plotly_white',  # White background for better visibility
            font=dict(color="white")  # White font for contrast
        )

        # Render the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    # Call the function to display combined charts
    combined_charts()
    # Factory by Trend Over Time
    trend_data = relevant_data.groupby(['Date', 'Factory']).agg(
        Total_Count=('Violation_Type', 'count')
    ).reset_index()

    # Create a new figure for the trend chart
    fig_trend = go.Figure()

    # Use a color palette for the Factory trends
    factory_colors = px.colors.qualitative.Vivid  # Vivid colors for factory trends

    for i, factory in enumerate(trend_data['Factory'].unique()):
        factory_data = trend_data[trend_data['Factory'] == factory]
        color = factory_colors[i % len(factory_colors)]  # Assign color to each factory
        fig_trend.add_trace(go.Scatter(
            x=factory_data['Date'],
            y=factory_data['Total_Count'],
            mode='lines+markers',
            name=str(factory),  # Ensure name is a string
            hoverinfo='text',
            text=factory_data['Total_Count'],
            line=dict(color=color, width=2),  # Set line color
            marker=dict(size=6, symbol='circle', color=color)  # Set marker color
        ))

    fig_trend.update_layout(
        title=f"{analysis_type} Trend Over Time by Factory",
        xaxis_title='Date',
        yaxis_title='Total Count',
        legend_title='Factory',
        template='plotly_white',  # Keep background white
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Employee by Over Time
    employee_data = relevant_data.groupby(['Date', 'Employee_Name']).agg(
        Total_Count=('Violation_Type', 'count')
    ).reset_index()

    # Create a new figure for the employee trend chart
    fig_employee = go.Figure()

    # Use a color palette for Employee trends
    employee_colors = px.colors.qualitative.Set2  # Soft colors for employee trends

    for i, employee in enumerate(employee_data['Employee_Name'].unique()):
        emp_data = employee_data[employee_data['Employee_Name'] == employee]
        color = employee_colors[i % len(employee_colors)]  # Assign color to each employee
        fig_employee.add_trace(go.Scatter(
            x=emp_data['Date'],
            y=emp_data['Total_Count'],
            mode='lines+markers',
            name=str(employee),  # Ensure name is a string
            hoverinfo='text',
            text=emp_data['Total_Count'],
            line=dict(color=color, width=2),  # Set line color
            marker=dict(size=6, symbol='circle', color=color)  # Set marker color
        ))

    fig_employee.update_layout(
        title=f"{analysis_type} Over Time by Employee",
        xaxis_title='Date',
        yaxis_title='Total Count',
        legend_title='Employee',
        template='plotly_white',  # Keep background white
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
    )

    st.plotly_chart(fig_employee, use_container_width=True)
elif view_type == "Unit":

    # Header for the analysis type by Unit
    st.header(f"{analysis_type} by Unit")

    # Define distinct colors for factories
    factory_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'  # Cyan
    ]

    # Factory-wise Violations/Compliance Chart
    st.subheader(f"{analysis_type} by Factory")

    factory_data = relevant_data.groupby(['Factory']).agg(
        Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
    ).reset_index()

    # Create a color map for factories
    factory_data['Color'] = factory_data.index.map(lambda x: factory_colors[x % len(factory_colors)])

    fig_factory = px.bar(factory_data, x='Factory', y='Total_Count',
                         title=f"{analysis_type} by Factory",
                         labels={
                             'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                         color='Color')  # Use the assigned color

    # Update layout for integer x-axis ticks
    fig_factory.update_layout(
        xaxis=dict(
            dtick=1,  # Set the tick interval to 1 for integer values
            tickmode='linear'  # Ensure ticks are linear
        )
    )

    st.plotly_chart(fig_factory, use_container_width=True)

    # Department-wise Violations/Compliance Chart
    st.subheader(f"{analysis_type} by Department")

    department_data = relevant_data.groupby(['Department']).agg(
        Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
    ).reset_index()

    # Define distinct colors for departments
    department_colors = [
        '#ffbb78',  # Light Orange
        '#98df8a',  # Light Green
        '#ff9896',  # Light Red
        '#c5b0d5',  # Light Purple
        '#f7b6d2',  # Light Pink
        '#c49c94',  # Light Brown
        '#f7f7f7',  # Light Gray
        '#dbdb8d',  # Light Olive
        '#9edae5',  # Light Cyan
        '#f3d9a4'  # Light Yellow
    ]

    # Create a color map for departments
    department_data['Color'] = department_data.index.map(lambda x: department_colors[x % len(department_colors)])

    fig_department = px.bar(department_data, x='Department', y='Total_Count',
                            title=f"{analysis_type} by Department",
                            labels={
                                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                            color='Color')  # Use the assigned color

    # Update layout for integer x-axis ticks
    fig_department.update_layout(
        xaxis=dict(
            dtick=1,  # Set the tick interval to 1 for integer values
            tickmode='linear'  # Ensure ticks are linear
        )
    )

    st.plotly_chart(fig_department, use_container_width=True)

    # Date-wise Violations/Compliance Chart
    st.subheader(f"{analysis_type} by Date")

    date_data = relevant_data.groupby(['Date']).agg(
        Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
    ).reset_index()

    fig_date = px.line(date_data, x='Date', y='Total_Count',
                       title=f"{analysis_type} Over Time",
                       labels={
                           'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                       color_discrete_sequence=px.colors.qualitative.Set1)  # Set a qualitative color scheme

    st.plotly_chart(fig_date, use_container_width=True)

    # Select relevant data based on the analysis type
    if analysis_type == "Violation":
        # For violation, we want records that are not compliant
        relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
        pie_title = "Violation Distribution by Type"

        # Pie chart for Violation Distribution
        fig = px.pie(relevant_data, names='Violation_Type', title=pie_title)

    else:
        # For compliance, we want to show the distribution of compliant vs non-compliant
        relevant_data = filtered_data.copy()
        relevant_data['Compliance_Status'] = relevant_data['Violation_Type'].apply(
            lambda x: 'Compliant' if x == 'Compliant' else 'Non-Compliant')
        pie_title = "Compliance vs Non-Compliance Distribution"

        # Pie chart for Compliance Distribution
        fig = px.pie(relevant_data, names='Compliance_Status', title=pie_title)

    # Display the pie chart
    st.plotly_chart(fig, use_container_width=True)

    # Filter relevant data based on the analysis type
    if analysis_type == "Violation":
        relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
        title = "Employee Violations"
    else:
        relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
        title = "Employee Compliance"

    # Group data by Employee to count occurrences
    employee_counts = relevant_data['Employee_Name'].value_counts().reset_index()
    employee_counts.columns = ['Employee_Name', 'Count']

    # Create a bar chart for Employee Compliance or Violations with distinct colors for each employee
    fig_employee_compliance = px.bar(employee_counts,
                                     x='Employee_Name',
                                     y='Count',
                                     title=title,
                                     labels={'Count': 'Total Count'},
                                     color='Employee_Name',  # Color by Employee_Name for distinct colors
                                     color_discrete_sequence=px.colors.qualitative.Plotly)  # Using a qualitative color palette

    # Update layout for better appearance
    fig_employee_compliance.update_layout(
        xaxis_title="Employee Name",
        yaxis_title="Count",
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
        font=dict(color="white")  # White font color for good contrast
    )

    # Display the bar chart
    st.plotly_chart(fig_employee_compliance, use_container_width=True)

    # Shift-wise Violations/Compliance Chart
    st.subheader(f"{analysis_type} by Shift")

    shift_data = relevant_data.groupby(['Shift']).agg(
        Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
    ).reset_index()

    # Create a bar chart for shifts using a vibrant color palette
    fig_shift = px.bar(shift_data, x='Shift', y='Total_Count',
                       title=f"{analysis_type} by Shift",
                       labels={
                           'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                       color='Shift',  # Color by Shift for distinct colors
                       color_discrete_sequence=px.colors.qualitative.Plotly)  # Using a qualitative color palette

    # Update layout for better appearance
    fig_shift.update_layout(
        xaxis_title="Shift",
        yaxis_title="Total Count",
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
        font=dict(color="white")  # White font color for good contrast
    )

    # Display the shift-wise chart
    st.plotly_chart(fig_shift, use_container_width=True)

    # Camera-wise Violations/Compliance Chart
    st.subheader(f"{analysis_type} by Camera")

    # Grouping data by Camera to count occurrences of violations
    camera_data = relevant_data.groupby(['Camera']).agg(
        Total_Violations=('Violation_Type', lambda x: (x != 'Compliant').sum()) if analysis_type == "Violation" else (
        'Employee_Name', 'count')
    ).reset_index()

    # Create a bar chart for total violations by camera with YlOrRd color scale
    fig_camera = px.bar(camera_data,
                        x='Camera',
                        y='Total_Violations',
                        title="Total Violations by Camera",
                        labels={'Total_Violations': 'Number of Violations'},
                        color='Total_Violations',  # Use Total_Violations for different colors
                        color_continuous_scale='YlOrRd')  # Yellow-Orange-Red color scale

    # Update layout for better appearance
    fig_camera.update_layout(
        xaxis_title="Camera",
        yaxis_title="Number of Violations",
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
        font=dict(color="black")  # Black font color for good contrast
    )

    # Display the chart
    st.plotly_chart(fig_camera, use_container_width=True)

    # Visualization Logic for Violations
    if analysis_type == "Violation":
        # Filter for Violations
        violation_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']

        # Create a Bar Chart for Violations
        if not violation_data.empty:
            violation_count = violation_data['Violation_Type'].value_counts().reset_index()
            violation_count.columns = ['Violation Type', 'Count']

            # Define distinct colors for violation types using qualitative colors
            fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                   title="Violation Counts",
                                   labels={'Count': 'Number of Violations'},
                                   color='Violation Type',  # Color by Violation Type for distinct colors
                                   color_discrete_sequence=px.colors.qualitative.Dark2)  # Darker color palette

            st.plotly_chart(fig_violation, use_container_width=True)
        else:
            st.write("No violation data available for the selected filters.")


elif view_type == "Red Zone":
    # Filter data based on user input
    if analysis_type == "Violation":
        st.subheader(f"Red Zone Alerts ({analysis_type} > 30%)")

        # Calculate violation rates for red zone monitoring
        red_zone_violations = filtered_data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
            lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

        red_zone_violation_alerts = red_zone_violations[
            red_zone_violations['Violation Rate'] > 30]  # More than 30% violations

        # Display the violation alerts as a dataframe
        st.dataframe(red_zone_violation_alerts)

        # Plot Red Zone Alerts as a Bar Chart
        fig_red_zone_violations = px.bar(red_zone_violation_alerts,
                                         x='Department',
                                         y='Violation Rate',
                                         color='Factory',
                                         title="Red Zone Violation Rates (More than 30% Violations)",
                                         labels={'Violation Rate': 'Violation Rate (%)'},
                                         color_discrete_sequence=px.colors.qualitative.Set2)  # Using distinct color set

        # Update layout for violation rates
        fig_red_zone_violations.update_layout(
            xaxis_title="Department",
            yaxis_title="Violation Rate (%)",
            yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
        )

        # Display the violation chart
        st.plotly_chart(fig_red_zone_violations, use_container_width=True)
    else:
        st.subheader("Red Zone Alerts (Compliance < 70%)")
        red_zone = filtered_data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
            lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

        red_zone_alerts = red_zone[red_zone['Compliance Rate'] < 70]

        # Display the red zone alerts as a dataframe
        st.dataframe(red_zone_alerts)

        # Plot Red Zone Alerts as a Bar Chart
        fig_red_zone = px.bar(red_zone_alerts,
                              x='Department',
                              y='Compliance Rate',
                              color='Factory',
                              title="Red Zone Compliance Rates (Less than 70%)",
                              labels={'Compliance Rate': 'Compliance Rate (%)'},
                              color_discrete_sequence=px.colors.qualitative.Set2)  # Using distinct color set

        # Update layout to ensure no decimal points on the x-axis
        fig_red_zone.update_layout(
            xaxis_title="Department",
            yaxis_title="Compliance Rate (%)",
            yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)  # Force x-axis to show integers only
        )

        # Display the chart
        st.plotly_chart(fig_red_zone, use_container_width=True)

elif view_type == "Monthly Rates and Prediction":
    st.subheader(f"{analysis_type} Monthly Rates and Prediction")
    # Create Line Chart for Monthly Rates and Prediction
    col1, col2, col3, col4 = st.columns(4)
    total_checks = filtered_data.shape[0]
    # Display Current Rate, Next Month Prediction, Total Checks, and Relevant Checks
    col1.metric(rate_label, f"{current_rate:.2f}%")
    col2.metric("Next Month Prediction", f"{next_month_prediction:.2f}%")
    col3.metric("Total Checks", total_checks)
    col4.metric("Relevant Checks", relevant_checks)

    # Combined Chart
    fig_combined = go.Figure()

    # Add Monthly Rates
    fig_combined.add_trace(go.Scatter(
        x=monthly_rate['Month'].astype(str),  # Ensuring x-axis is in string format
        y=monthly_rate['Rate'],
        mode='lines+markers',
        name='Monthly Rate',
        line=dict(color='royalblue'),  # A pleasing blue for the monthly rate line
        marker=dict(size=8, color='lightblue')  # Lighter blue for markers
    ))

    # Prepare the next month label
    next_month_label = pd.to_datetime(monthly_rate['Month'].iloc[-1]).to_period(
        'M').to_timestamp() + pd.offsets.MonthEnd(1)

    # Add Predicted Rate
    fig_combined.add_trace(go.Scatter(
        x=[*monthly_rate['Month'].astype(str), next_month_label.strftime('%Y-%m')],
        y=[*monthly_rate['Rate'], next_month_prediction],
        mode='lines+markers+text',
        name='Predicted Rate',
        text=[*[''] * len(monthly_rate), f"{next_month_prediction:.2f}%"],
        textposition='top center',
        line=dict(color='orange', dash='dash'),  # Orange for predicted rate
        marker=dict(size=8, color='orange')  # Orange for markers
    ))

    # Update layout
    fig_combined.update_layout(
        title=f"{analysis_type} Rate Over Time and Prediction",
        xaxis_title='Month',
        yaxis_title='Rate (%)',
        xaxis=dict(tickvals=[*monthly_rate['Month'].astype(str), next_month_label.strftime('%Y-%m')]),
        showlegend=True,
    )



    # Display Combined Chart and Table
    st.plotly_chart(fig_combined, use_container_width=True)


    # Create Difference Chart
    if relevant_data.shape[0] > 0:
        difference = next_month_prediction - current_rate
        fig_difference = go.Figure()

        # Add Difference
        fig_difference.add_trace(go.Bar(
            x=['Current Rate', 'Next Month Prediction'],
            y=[current_rate, next_month_prediction],
            name='Rates',
            marker_color=['royalblue', 'orange'],  # Use blue and orange for bars
        ))

        # Add Line for Difference
        fig_difference.add_trace(go.Scatter(
            x=['Current Rate', 'Next Month Prediction'],
            y=[current_rate, next_month_prediction],
            mode='lines+text',
            name='Difference',
            text=[f"{difference:.2f}%" if difference > 0 else f"{-difference:.2f}% down", ''],
            textposition='top center',
            line=dict(color='red', width=2)  # Red line for difference
        ))

        # Update layout for difference chart
        fig_difference.update_layout(
            title='Current vs. Predicted Rate Difference',
            xaxis_title='Rate Type',
            yaxis_title='Rate (%)',
            showlegend=True,
        )

        # Display Difference Chart
        st.plotly_chart(fig_difference, use_container_width=True)



elif view_type == "Target":


    
    target_compliance_rate = 90  # Target compliance rate for compliance analysis
    target_violation_rate = 10  # Target violation rate for violation analysis

    # Visualization Logic
    if analysis_type == "Violation":
        st.subheader("Violation Target is 10%")
        # Filter for Violations
        violation_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']

        # Create a Difference Chart
        if not violation_data.empty:
            violation_count = violation_data['Violation_Type'].value_counts().reset_index()
            violation_count.columns = ['Violation Type', 'Count']

            # Current Violation Rate Calculation
            current_violation_rate = (violation_count['Count'].sum() / filtered_data.shape[0]) * 100 if \
                filtered_data.shape[0] > 0 else 0

            # Create a Difference Chart
            fig_difference = go.Figure()
            fig_difference.add_trace(go.Bar(
                x=['Current Rate', 'Target Rate'],
                y=[current_violation_rate, target_violation_rate],
                name='Rates',
                marker_color=['darkblue', 'orange'],  # Dark blue for current rate, orange for target
            ))

            # Add Line for Difference
            fig_difference.add_trace(go.Scatter(
                x=['Current Rate', 'Target Rate'],
                y=[current_violation_rate, target_violation_rate],
                mode='lines+text',
                name='Difference',
                text=[f"{current_violation_rate:.2f}%", f"{target_violation_rate:.2f}%"],
                textposition='top center',
                line=dict(color='red', width=2)  # Keep the line color red
            ))

            # Update layout for difference chart
            fig_difference.update_layout(
                title='Current vs. Target Violation Rate',
                xaxis_title='Rate Type',
                yaxis_title='Rate (%)',
                showlegend=True,
                plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background for better appearance
            )

            # Display Difference Chart above the Bar Chart
            st.plotly_chart(fig_difference, use_container_width=True)

            # Create a Bar Chart for Violations
            fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                   title="Violation Counts",
                                   labels={'Count': 'Number of Violations'},
                                   color='Violation Type',  # Use the Violation Type for color mapping
                                   color_discrete_sequence=px.colors.qualitative.Dark2)  # Darker color palette

            # Update layout for violation chart
            fig_violation.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
                font=dict(color="white"),  # White font color for good contrast
            )

            # Display Bar Chart for Violations
            st.plotly_chart(fig_violation, use_container_width=True)

        else:
            st.write("No violation data available for the selected filters.")

    elif analysis_type == "Compliance":
        st.subheader("Compliance Target is 90%")
        # Filter for Compliance
        compliance_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']

        # Create a Difference Chart for Compliance
        if not compliance_data.empty:
            compliance_count = compliance_data['Violation_Type'].value_counts().reset_index()
            compliance_count.columns = ['Compliance Type', 'Count']

            # Current Compliance Rate Calculation
            current_compliance_rate = (compliance_count['Count'].sum() / filtered_data.shape[0]) * 100 if \
                filtered_data.shape[0] > 0 else 0

            # Create a Difference Chart for Compliance
            fig_difference_compliance = go.Figure()
            fig_difference_compliance.add_trace(go.Bar(
                x=['Current Rate', 'Target Rate'],
                y=[current_compliance_rate, target_compliance_rate],
                name='Rates',
                marker_color=['darkgreen', 'orange'],  # Dark green for current rate, orange for target
            ))

            # Add Line for Difference
            fig_difference_compliance.add_trace(go.Scatter(
                x=['Current Rate', 'Target Rate'],
                y=[current_compliance_rate, target_compliance_rate],
                mode='lines+text',
                name='Difference',
                text=[f"{current_compliance_rate:.2f}%", f"{target_compliance_rate:.2f}%"],
                textposition='top center',
                line=dict(color='red', width=2)  # Keep the line color red
            ))

            # Update layout for compliance difference chart
            fig_difference_compliance.update_layout(
                title='Current vs. Target Compliance Rate',
                xaxis_title='Rate Type',
                yaxis_title='Rate (%)',
                showlegend=True,
                plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background for better appearance
            )

            # Display Difference Chart above the Bar Chart
            st.plotly_chart(fig_difference_compliance, use_container_width=True)

            # Create a Bar Chart for Compliance
            fig_compliance = px.bar(compliance_count, x='Compliance Type', y='Count',
                                    title="Compliance Counts",
                                    labels={'Count': 'Number of Compliant Cases'},
                                    color='Compliance Type',  # Use the Compliance Type for color mapping
                                    color_discrete_sequence=px.colors.qualitative.Dark2)  # Darker color palette

            # Update layout for compliance chart
            fig_compliance.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
                font=dict(color="white"),  # White font color for good contrast
            )

            # Display Bar Chart for Compliance
            st.plotly_chart(fig_compliance, use_container_width=True)

        else:
            st.write("No compliance data available for the selected filters.")
