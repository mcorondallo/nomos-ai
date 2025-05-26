import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.units import inch
import datetime
import xlsxwriter

def load_dummy_data(n_samples=200):
    """
    Generate dummy data for development and testing
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        DataFrame: Dummy historical payment data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create empty dataframe
    data = pd.DataFrame()
    
    # Generate payment histories
    data['days_overdue'] = np.random.randint(0, 90, size=n_samples)
    data['previous_late_payments'] = np.random.randint(0, 6, size=n_samples)
    data['average_days_late'] = np.random.randint(0, 45, size=n_samples)
    
    # Generate lease information
    data['lease_count'] = np.random.randint(1, 5, size=n_samples)
    data['lease_amount'] = np.random.randint(500, 5000, size=n_samples)
    data['property_size_sqm'] = np.random.randint(50, 500, size=n_samples)
    
    # Generate categorical features
    payment_methods = ['bank_transfer', 'card']
    data['payment_method'] = np.random.choice(payment_methods, size=n_samples)
    
    countries = [
        'France', 'Germany', 'Spain', 'Italy', 'UK', 
        'Netherlands', 'Belgium', 'Portugal', 'Austria', 'Switzerland'
    ]
    data['country'] = np.random.choice(countries, size=n_samples)
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    data['region'] = np.random.choice(regions, size=n_samples)
    
    # Generate target variable (was payment made?)
    # More likely to be paid if:
    # - fewer days overdue
    # - fewer previous late payments
    # - lower average days late
    # - higher lease count (more history)
    
    # Calculate a base probability
    p_base = 0.7
    
    # Adjust based on features (simplified model for dummy data)
    p_adjust = (
        -0.005 * data['days_overdue'] +
        -0.03 * data['previous_late_payments'] +
        -0.003 * data['average_days_late'] +
        0.05 * data['lease_count']
    )
    
    # Calculate final probability, ensuring it's between 0 and 1
    p_final = np.clip(p_base + p_adjust, 0.1, 0.9)
    
    # Generate binary outcome based on probability
    data['was_paid'] = np.random.binomial(1, p_final)
    
    return data

def generate_sample_template(prediction_only=False):
    """
    Generate a sample CSV template for data input
    
    Args:
        prediction_only (bool): If True, exclude target variable
        
    Returns:
        BytesIO: CSV template as bytes buffer
    """
    buffer = io.StringIO()
    
    # Define columns based on whether this is for prediction or training
    columns = [
        'days_overdue', 'previous_late_payments', 'average_days_late',
        'lease_count', 'lease_amount', 'property_size_sqm',
        'payment_method', 'country', 'region'
    ]
    
    if not prediction_only:
        columns.append('was_paid')
    
    # Create empty DataFrame with columns
    template_df = pd.DataFrame(columns=columns)
    
    # Add 3 sample rows with example data
    sample_data = [
        {
            'days_overdue': 15, 
            'previous_late_payments': 1, 
            'average_days_late': 10,
            'lease_count': 2, 
            'lease_amount': 2000, 
            'property_size_sqm': 150,
            'payment_method': 'bank_transfer', 
            'country': 'France', 
            'region': 'Central'
        },
        {
            'days_overdue': 45, 
            'previous_late_payments': 3, 
            'average_days_late': 30,
            'lease_count': 1, 
            'lease_amount': 1500, 
            'property_size_sqm': 100,
            'payment_method': 'card', 
            'country': 'Germany', 
            'region': 'East'
        },
        {
            'days_overdue': 5, 
            'previous_late_payments': 0, 
            'average_days_late': 0,
            'lease_count': 3, 
            'lease_amount': 3000, 
            'property_size_sqm': 200,
            'payment_method': 'bank_transfer', 
            'country': 'Spain', 
            'region': 'South'
        }
    ]
    
    for sample in sample_data:
        if not prediction_only:
            # Add was_paid with logical values based on other features
            if sample['days_overdue'] < 30 and sample['previous_late_payments'] < 2:
                sample['was_paid'] = 1
            else:
                sample['was_paid'] = 0
        
        template_df = pd.concat([template_df, pd.DataFrame([sample])], ignore_index=True)
    
    # Write to buffer
    template_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer.getvalue()

def generate_excel_report(results_df, benchmark=0.75, include_charts=True, include_detailed=True):
    """
    Generate an Excel report with prediction results and visualizations
    
    Args:
        results_df (DataFrame): Prediction results
        benchmark (float): Benchmark value for comparison
        include_charts (bool): Whether to include charts
        include_detailed (bool): Whether to include detailed data tables
        
    Returns:
        BytesIO: Excel report as bytes buffer
    """
    buffer = io.BytesIO()
    
    # Create Excel writer
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Add title formats
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#0066cc',
            'font_color': 'white'
        })
        
        subtitle_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'align': 'left',
            'valign': 'vcenter',
            'bg_color': '#f0f2f6'
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 11,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#dddddd',
            'border': 1
        })
        
        percent_format = workbook.add_format({'num_format': '0.00%'})
        
        # Summary worksheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.set_column('A:A', 25)
        summary_sheet.set_column('B:B', 15)
        
        # Add title
        summary_sheet.merge_range('A1:B1', 'Nomos AI Payment Prediction Report', title_format)
        summary_sheet.write('A2', f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', workbook.add_format({'italic': True}))
        
        # Add summary statistics
        summary_sheet.merge_range('A4:B4', 'Summary Statistics', subtitle_format)
        
        # Calculate summary data
        avg_probability = results_df['payment_probability'].mean()
        median_probability = results_df['payment_probability'].median()
        min_probability = results_df['payment_probability'].min()
        max_probability = results_df['payment_probability'].max()
        
        # Count by risk category
        risk_counts = results_df['risk_category'].value_counts().to_dict()
        high_risk = risk_counts.get('High Risk', 0)
        medium_risk = risk_counts.get('Medium Risk', 0)
        low_risk = risk_counts.get('Low Risk', 0)
        
        # Add summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Number of Records', len(results_df)],
            ['Average Payment Probability', avg_probability],
            ['Median Payment Probability', median_probability],
            ['Minimum Probability', min_probability],
            ['Maximum Probability', max_probability],
            ['Benchmark', benchmark],
            ['Above Benchmark', (results_df['payment_probability'] >= benchmark).sum()],
            ['Below Benchmark', (results_df['payment_probability'] < benchmark).sum()],
            ['High Risk Count', high_risk],
            ['Medium Risk Count', medium_risk],
            ['Low Risk Count', low_risk]
        ]
        
        # Write summary table
        summary_sheet.write_row('A6', summary_data[0], header_format)
        for i, row in enumerate(summary_data[1:]):
            summary_sheet.write('A' + str(i + 7), row[0])
            
            # Format probability values as percentages
            if 'Probability' in row[0] or row[0] == 'Benchmark':
                summary_sheet.write('B' + str(i + 7), row[1], percent_format)
            else:
                summary_sheet.write('B' + str(i + 7), row[1])
        
        # Add charts if requested
        if include_charts:
            # Risk distribution pie chart
            risk_sheet = workbook.add_worksheet('Risk Analysis')
            risk_sheet.set_column('A:B', 15)
            
            risk_sheet.merge_range('A1:C1', 'Risk Category Distribution', title_format)
            
            # Prepare risk data
            risk_data = [
                ['Risk Category', 'Count', 'Percentage'],
                ['High Risk', high_risk, high_risk/len(results_df)],
                ['Medium Risk', medium_risk, medium_risk/len(results_df)],
                ['Low Risk', low_risk, low_risk/len(results_df)]
            ]
            
            # Write risk data
            risk_sheet.write_row('A3', risk_data[0], header_format)
            for i, row in enumerate(risk_data[1:]):
                risk_sheet.write('A' + str(i + 4), row[0])
                risk_sheet.write('B' + str(i + 4), row[1])
                risk_sheet.write('C' + str(i + 4), row[2], percent_format)
            
            # Create pie chart for risk distribution
            pie_chart = workbook.add_chart({'type': 'pie'})
            pie_chart.add_series({
                'name': 'Risk Distribution',
                'categories': ['Risk Analysis', 3, 0, 5, 0],
                'values': ['Risk Analysis', 3, 1, 5, 1],
                'points': [
                    {'fill': {'color': '#FF0000'}},  # High Risk - Red
                    {'fill': {'color': '#FFA500'}},  # Medium Risk - Orange
                    {'fill': {'color': '#00AA00'}}   # Low Risk - Green
                ],
                'data_labels': {'percentage': True}
            })
            
            pie_chart.set_title({'name': 'Risk Category Distribution'})
            pie_chart.set_style(10)  # Use a cleaner style
            risk_sheet.insert_chart('D3', pie_chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Add country analysis if available
            if 'country' in results_df.columns:
                country_sheet = workbook.add_worksheet('Geographic Analysis')
                country_sheet.set_column('A:D', 15)
                
                country_sheet.merge_range('A1:D1', 'Payment Probability by Country', title_format)
                
                # Group by country
                country_data = results_df.groupby('country').agg({
                    'payment_probability': ['mean', 'median', 'count']
                }).reset_index()
                
                # Flatten column names
                country_data.columns = ['Country', 'Mean Probability', 'Median Probability', 'Count']
                
                # Write country data
                columns = ['Country', 'Mean Probability', 'Median Probability', 'Count']
                country_sheet.write_row('A3', columns, header_format)
                
                for i, row in enumerate(country_data.itertuples(index=False)):
                    country_sheet.write('A' + str(i + 4), row[0])
                    country_sheet.write('B' + str(i + 4), row[1], percent_format)
                    country_sheet.write('C' + str(i + 4), row[2], percent_format)
                    country_sheet.write('D' + str(i + 4), row[3])
                
                # Create bar chart for country analysis
                bar_chart = workbook.add_chart({'type': 'column'})
                bar_chart.add_series({
                    'name': 'Mean Payment Probability',
                    'categories': ['Geographic Analysis', 3, 0, 3 + len(country_data) - 1, 0],
                    'values': ['Geographic Analysis', 3, 1, 3 + len(country_data) - 1, 1],
                    'data_labels': {'value': True, 'num_format': '0.0%'}
                })
                
                bar_chart.set_title({'name': 'Mean Payment Probability by Country'})
                bar_chart.set_x_axis({'name': 'Country'})
                bar_chart.set_y_axis({'name': 'Probability', 'num_format': '0.0%'})
                
                country_sheet.insert_chart('F3', bar_chart, {'x_scale': 1.5, 'y_scale': 1.5})
        
        # Add detailed results if requested
        if include_detailed:
            results_sheet = workbook.add_worksheet('Detailed Results')
            
            # Format columns appropriately
            for i, col in enumerate(results_df.columns):
                # Auto-fit column width based on content
                max_len = max(
                    len(col), 
                    results_df[col].astype(str).map(len).max() + 2
                )
                results_sheet.set_column(i, i, max_len)
            
            # Write header row
            for i, col in enumerate(results_df.columns):
                results_sheet.write(0, i, col, header_format)
            
            # Write data rows
            for i, row in enumerate(results_df.itertuples(index=False)):
                for j, val in enumerate(row):
                    cell_format = None
                    # Apply percentage format for probability columns
                    if 'probability' in results_df.columns[j] or 'bound' in results_df.columns[j]:
                        cell_format = percent_format
                    
                    results_sheet.write(i + 1, j, val, cell_format)
    
    # Return the buffer
    buffer.seek(0)
    return buffer

def generate_pdf_report(results_df, benchmark=0.75, include_region=True, include_confidence=False):
    """
    Generate a PDF report with prediction results and comparisons
    
    Args:
        results_df (DataFrame): Prediction results
        benchmark (float): Benchmark value for comparison
        include_region (bool): Whether to include regional analysis
        include_confidence (bool): Whether to include confidence intervals
        
    Returns:
        BytesIO: PDF report as bytes buffer
    """
    buffer = io.BytesIO()
    
    # Create the document
    now = datetime.datetime.now()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                         rightMargin=72, leftMargin=72,
                         topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    centered_style = ParagraphStyle(
        'centered',
        parent=styles['Normal'],
        alignment=1,
    )
    
    # Add title
    elements.append(Paragraph('Nomos AI Payment Prediction Report', title_style))
    elements.append(Spacer(1, 12))
    
    # Add date
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {date_string}", normal_style))
    elements.append(Spacer(1, 24))
    
    # Add summary statistics
    elements.append(Paragraph('Summary Statistics', heading_style))
    elements.append(Spacer(1, 12))
    
    # Calculate summary data
    avg_probability = results_df['payment_probability'].mean()
    median_probability = results_df['payment_probability'].median()
    min_probability = results_df['payment_probability'].min()
    max_probability = results_df['payment_probability'].max()
    
    # Prepare summary data
    summary_data = [
        ['Metric', 'Value'],
        ['Records Analyzed', str(len(results_df))],
        ['Average Payment Probability', f"{avg_probability:.2%}"],
        ['Median Payment Probability', f"{median_probability:.2%}"],
        ['Minimum Probability', f"{min_probability:.2%}"],
        ['Maximum Probability', f"{max_probability:.2%}"],
        ['Benchmark', f"{benchmark:.2%}"],
        ['Above Benchmark', str((results_df['payment_probability'] >= benchmark).sum())],
        ['Below Benchmark', str((results_df['payment_probability'] < benchmark).sum())]
    ]
    
    # Calculate risk categories if they exist
    if 'risk_category' in results_df.columns:
        risk_counts = results_df['risk_category'].value_counts().to_dict()
        summary_data.extend([
            ['High Risk Count', str(risk_counts.get('High Risk', 0))],
            ['Medium Risk Count', str(risk_counts.get('Medium Risk', 0))],
            ['Low Risk Count', str(risk_counts.get('Low Risk', 0))]
        ])
    
    # Create summary table
    summary_table = Table(summary_data, colWidths=[250, 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (1, -1), 1, colors.black)
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 24))
    
    # Add risk distribution pie chart if risk_category exists
    if 'risk_category' in results_df.columns:
        elements.append(Paragraph('Risk Distribution', heading_style))
        elements.append(Spacer(1, 12))
        
        # Get risk counts
        risk_counts = results_df['risk_category'].value_counts()
        
        # Create drawing for pie chart
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 100
        pie.y = 0
        pie.width = 200
        pie.height = 200
        
        # Add data to pie chart
        pie.data = list(risk_counts.values)
        pie.labels = list(risk_counts.index)
        
        # Set colors for risk categories
        risk_colors = {
            'High Risk': colors.red,
            'Medium Risk': colors.orange,
            'Low Risk': colors.green
        }
        
        # Apply colors based on available risk categories
        pie_colors = [risk_colors.get(category, colors.blue) for category in risk_counts.index]
        pie.slices.strokeWidth = 0.5
        pie.slices.strokeColor = colors.white
        
        for i, color in enumerate(pie_colors):
            pie.slices[i].fillColor = color
        
        drawing.add(pie)
        elements.append(drawing)
        elements.append(Spacer(1, 24))
    
    # Add country/region analysis if include_region is True
    if include_region and 'country' in results_df.columns:
        elements.append(Paragraph('Geographic Analysis', heading_style))
        elements.append(Spacer(1, 12))
        
        # Group by country and calculate stats
        country_stats = results_df.groupby('country')['payment_probability'].agg(['mean', 'count']).reset_index()
        country_stats.columns = ['Country', 'Mean Probability', 'Count']
        country_stats['Mean Probability'] = country_stats['Mean Probability'].map(lambda x: f"{x:.2%}")
        
        # Create country table
        country_data = [
            ['Country', 'Mean Payment Probability', 'Count']
        ]
        for _, row in country_stats.iterrows():
            country_data.append([row['Country'], row['Mean Probability'], str(row['Count'])])
        
        country_table = Table(country_data, colWidths=[150, 150, 100])
        country_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (2, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (2, 0), 12),
            ('BOTTOMPADDING', (0, 0), (2, 0), 8),
            ('BACKGROUND', (0, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (2, -1), 1, colors.black)
        ]))
        
        elements.append(country_table)
        elements.append(Spacer(1, 24))
        
        # Add region analysis if available
        if 'region' in results_df.columns:
            # Group by region and calculate stats
            region_stats = results_df.groupby(['country', 'region'])['payment_probability'].agg(['mean', 'count']).reset_index()
            region_stats.columns = ['Country', 'Region', 'Mean Probability', 'Count']
            region_stats['Mean Probability'] = region_stats['Mean Probability'].map(lambda x: f"{x:.2%}")
            
            # Create region table
            region_data = [
                ['Country', 'Region', 'Mean Payment Probability', 'Count']
            ]
            for _, row in region_stats.iterrows():
                region_data.append([row['Country'], row['Region'], row['Mean Probability'], str(row['Count'])])
            
            region_table = Table(region_data, colWidths=[100, 100, 150, 100])
            region_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (3, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (3, 0), 12),
                ('BOTTOMPADDING', (0, 0), (3, 0), 8),
                ('BACKGROUND', (0, 1), (3, -1), colors.beige),
                ('GRID', (0, 0), (3, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph('Regional Analysis', heading2_style))
            elements.append(Spacer(1, 12))
            elements.append(region_table)
            elements.append(Spacer(1, 24))
    
    # Add confidence intervals if requested
    if include_confidence and 'lower_bound' in results_df.columns and 'upper_bound' in results_df.columns:
        elements.append(Paragraph('Confidence Intervals Analysis', heading_style))
        elements.append(Spacer(1, 12))
        
        # Calculate average bounds
        avg_lower = results_df['lower_bound'].mean()
        avg_upper = results_df['upper_bound'].mean()
        avg_range = avg_upper - avg_lower
        
        confidence_text = f"""The average confidence interval for payment predictions ranges from {avg_lower:.2%} to {avg_upper:.2%}, 
        with an average range of {avg_range:.2%}. Narrower confidence intervals indicate higher prediction certainty."""
        
        elements.append(Paragraph(confidence_text, normal_style))
        elements.append(Spacer(1, 12))
        
        # Create data for confidence ranges
        confidence_ranges = [
            ['Range Width', 'Count', 'Percentage'],
            ['Narrow (<10%)', 0, 0],
            ['Medium (10-20%)', 0, 0],
            ['Wide (>20%)', 0, 0]
        ]
        
        # Calculate counts for each range category
        ranges = results_df['upper_bound'] - results_df['lower_bound']
        narrow = (ranges < 0.1).sum()
        medium = ((ranges >= 0.1) & (ranges <= 0.2)).sum()
        wide = (ranges > 0.2).sum()
        total = len(results_df)
        
        confidence_ranges[1][1] = narrow
        confidence_ranges[1][2] = f"{narrow/total:.1%}"
        confidence_ranges[2][1] = medium
        confidence_ranges[2][2] = f"{medium/total:.1%}"
        confidence_ranges[3][1] = wide
        confidence_ranges[3][2] = f"{wide/total:.1%}"
        
        confidence_table = Table(confidence_ranges, colWidths=[150, 100, 100])
        confidence_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (2, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (2, 0), 8),
            ('BACKGROUND', (0, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (2, -1), 1, colors.black)
        ]))
        
        elements.append(confidence_table)
        elements.append(Spacer(1, 24))
    
    # Add payment patterns analysis if available features exist
    if 'payment_method' in results_df.columns and 'days_overdue' in results_df.columns:
        elements.append(Paragraph('Payment Patterns Analysis', heading_style))
        elements.append(Spacer(1, 12))
        
        # Analyze payment patterns by payment method
        if 'payment_method' in results_df.columns:
            payment_method_stats = results_df.groupby('payment_method')['payment_probability'].agg(['mean', 'count']).reset_index()
            payment_method_stats.columns = ['Payment Method', 'Mean Probability', 'Count']
            payment_method_stats['Mean Probability'] = payment_method_stats['Mean Probability'].map(lambda x: f"{x:.2%}")
            
            # Create payment method table
            payment_method_data = [
                ['Payment Method', 'Mean Payment Probability', 'Count']
            ]
            for _, row in payment_method_stats.iterrows():
                payment_method_data.append([row['Payment Method'], row['Mean Probability'], str(row['Count'])])
            
            payment_method_table = Table(payment_method_data, colWidths=[150, 150, 100])
            payment_method_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (2, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (2, 0), 12),
                ('BOTTOMPADDING', (0, 0), (2, 0), 8),
                ('BACKGROUND', (0, 1), (2, -1), colors.beige),
                ('GRID', (0, 0), (2, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph('Payment Method Analysis', heading2_style))
            elements.append(Spacer(1, 12))
            elements.append(payment_method_table)
            elements.append(Spacer(1, 24))
        
        # Analyze days overdue impact
        if 'days_overdue' in results_df.columns:
            # Create bins for days overdue
            bins = [0, 15, 30, 60, 90, float('inf')]
            labels = ['0-15 days', '16-30 days', '31-60 days', '61-90 days', '90+ days']
            results_df['overdue_category'] = pd.cut(results_df['days_overdue'], bins=bins, labels=labels)
            
            overdue_stats = results_df.groupby('overdue_category')['payment_probability'].agg(['mean', 'count']).reset_index()
            overdue_stats.columns = ['Days Overdue', 'Mean Probability', 'Count']
            overdue_stats['Mean Probability'] = overdue_stats['Mean Probability'].map(lambda x: f"{x:.2%}")
            
            # Create overdue table
            overdue_data = [
                ['Days Overdue', 'Mean Payment Probability', 'Count']
            ]
            for _, row in overdue_stats.iterrows():
                overdue_data.append([row['Days Overdue'], row['Mean Probability'], str(row['Count'])])
            
            overdue_table = Table(overdue_data, colWidths=[150, 150, 100])
            overdue_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (2, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (2, 0), 12),
                ('BOTTOMPADDING', (0, 0), (2, 0), 8),
                ('BACKGROUND', (0, 1), (2, -1), colors.beige),
                ('GRID', (0, 0), (2, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph('Days Overdue Analysis', heading2_style))
            elements.append(Spacer(1, 12))
            elements.append(overdue_table)
            elements.append(Spacer(1, 24))
    
    # Detailed records (top high and low probability)
    elements.append(Paragraph('Sample Records', heading_style))
    elements.append(Spacer(1, 12))
    
    # Get top 5 highest and lowest probability records
    top_high = results_df.sort_values('payment_probability', ascending=False).head(5)
    top_low = results_df.sort_values('payment_probability', ascending=True).head(5)
    
    # Create sample data
    sample_data = [
        ['Record ID', 'Payment Probability', 'Risk Category']
    ]
    
    # Add high probability samples
    for i, row in enumerate(top_high.itertuples()):
        record_id = getattr(row, 'Index', i) if hasattr(row, 'Index') else i
        probability = getattr(row, 'payment_probability')
        risk = getattr(row, 'risk_category', 'N/A') if hasattr(row, 'risk_category') else 'N/A'
        sample_data.append([f"HIGH-{record_id}", f"{probability:.2%}", risk])
    
    # Add low probability samples
    for i, row in enumerate(top_low.itertuples()):
        record_id = getattr(row, 'Index', i) if hasattr(row, 'Index') else i
        probability = getattr(row, 'payment_probability')
        risk = getattr(row, 'risk_category', 'N/A') if hasattr(row, 'risk_category') else 'N/A'
        sample_data.append([f"LOW-{record_id}", f"{probability:.2%}", risk])
    
    sample_table = Table(sample_data, colWidths=[100, 150, 150])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (2, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (2, 0), 12),
        ('BOTTOMPADDING', (0, 0), (2, 0), 8),
        ('BACKGROUND', (0, 1), (2, 5), colors.lightgreen),  # High probability rows
        ('BACKGROUND', (0, 6), (2, -1), colors.salmon),      # Low probability rows
        ('GRID', (0, 0), (2, -1), 1, colors.black)
    ]))
    
    elements.append(sample_table)
    
    # Add footer
    elements.append(Spacer(1, 36))
    elements.append(Paragraph('Generated by Nomos AI - Arrears Prediction System', centered_style))
    
    # Build the PDF document
    doc.build(elements)
    
    elements.append(Spacer(1, 12))
    
    # Analyze payment patterns by payment method
    if 'payment_method' in results_df.columns:
        payment_method_stats = results_df.groupby('payment_method')['payment_probability'].agg(['mean', 'count']).reset_index()
        payment_method_stats.columns = ['Payment Method', 'Mean Probability', 'Count']
        payment_method_stats['Mean Probability'] = payment_method_stats['Mean Probability'].map(lambda x: f"{x:.2%}")
    elements.append(Paragraph("Benchmark Comparison", subtitle_style))
    
    benchmark_text = f"The average payment probability ({avg_probability:.2%}) "
    if avg_probability > benchmark:
        benchmark_text += f"is {avg_probability - benchmark:.2%} higher than the benchmark ({benchmark:.2%}), indicating better than expected payment likelihood."
    elif avg_probability < benchmark:
        benchmark_text += f"is {benchmark - avg_probability:.2%} lower than the benchmark ({benchmark:.2%}), indicating worse than expected payment likelihood."
    else:
        benchmark_text += f"is equal to the benchmark ({benchmark:.2%})."
        
    elements.append(Paragraph(benchmark_text, normal_style))
    elements.append(Spacer(1, 20))
    
    # Recommendations section
    elements.append(Paragraph("Recommendations", subtitle_style))
    
    # Generate some simple recommendations based on the results
    recommendations = []
    
    if high_risk > (medium_risk + low_risk):
        recommendations.append("A large portion of predictions are high risk. Consider reviewing collection strategies and payment terms.")
    
    if avg_probability < benchmark:
        recommendations.append(f"Overall payment probability is below benchmark. Further investigation into payment factors is recommended.")
    
    if high_risk > 0:
        recommendations.append(f"Prioritize following up with the {high_risk} high-risk cases identified in this analysis.")
    
    # Add recommendations as bullet points
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", normal_style))
    
    elements.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer = "This report is generated based on predictive modeling and historical data analysis. " \
                "Actual payment outcomes may vary. The model should be periodically retrained with " \
                "new data to maintain accuracy."
    elements.append(Paragraph(disclaimer, normal_style))
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer
