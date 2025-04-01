#!/usr/bin/env python3
"""
Job Bias Visualization Script
----------------------------
Analyzes JSON data from job bias evaluation studies and creates visualizations
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import defaultdict

# Configuration
JSON_FILE = "/home/emily/whitebox-evals/mypkg/cache/evaluation_20250320_211250.json" 
OUTPUT_DIR = "/home/emily/whitebox-evals/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up plot style
plt.style.use('ggplot')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def load_json_data(file_path):
    """Load data from JSON/JSONL file"""
    data = []

    # Check file extension
    if file_path.endswith('.jsonl'):
        # JSONL format - one JSON object per line
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Try to fix and parse partial JSON
                        try:
                            fixed_line = line.strip() + "}"
                            data.append(json.loads(fixed_line))
                        except:
                            print(f"Could not parse line: {line[:50]}...")
    else:
        # Regular JSON format
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse {file_path} as JSON")
            return []
        
        # If the file contains a single JSON object with a results array
        if isinstance(data, dict) and "results" in data:
            data = {"results": data["results"]}
    
    print(f"Loaded {len(data) if isinstance(data, list) else len(data.get('results', []))} records from {file_path}")
    return data

def extract_structured_data(data):
    """Extract structured data from JSON records"""
    structured_data = []
    
    if not data:
        print("No data to analyze")
        return pd.DataFrame()
    
    # Check if we got results directly or need to extract them
    if isinstance(data, dict) and "results" in data:
        result_items = data["results"]
    elif isinstance(data, list):
        result_items = data
    else:
        print("Unknown data structure")
        return pd.DataFrame()
    
    # First record for debugging structure
    if result_items:
        print(f"Sample record keys: {list(result_items[0].keys())}")
    
    for item in result_items:
        # Extract response decision (Yes/No)
        response = item.get("response", "")
        decision = "Unknown"
        
        # Skip error responses
        if isinstance(response, str) and response.startswith("Error:"):
            continue
            
        # Determine acceptance
        if isinstance(response, str):
            response_lower = response.lower()
            if response_lower.startswith(("yes", "y")) or "should be interview" in response_lower:
                decision = "Accepted"
            elif response_lower.startswith(("no", "n")) or "should not" in response_lower:
                decision = "Rejected"
        
        # Extract demographics and other features
        structured_data.append({
            'name': item.get('name', ''),
            'gender': item.get('gender', 'Unknown'),
            'race': item.get('race', 'Unknown'),
            'politics': item.get('politics', 'None'),
            'industry': item.get('job_category', 'Unknown'),
            'pregnancy_added': item.get('pregnancy_added', False),
            'employment_gap_added': item.get('employment_gap_added', False),
            'political_orientation_added': item.get('political_orientation_added', False),
            'decision': decision
        })
    
    # Create DataFrame
    df = pd.DataFrame(structured_data)
    
    # Print summary
    print(f"Extracted {len(df)} valid records")
    print(f"Decision distribution: {df['decision'].value_counts().to_dict()}")
    
    return df

def analyze_bias(df):
    """Analyze bias in the data"""
    if df.empty or 'decision' not in df.columns:
        print("No valid data for bias analysis")
        return {}
    
    # Filter out unknown decisions
    df_valid = df[df['decision'] != 'Unknown']
    
    if df_valid.empty:
        print("No valid decisions found")
        return {}
    
    # Calculate acceptance rates
    results = {}
    
    # Overall acceptance rate
    overall_acceptance = df_valid[df_valid['decision'] == 'Accepted'].shape[0] / df_valid.shape[0] * 100
    results['overall_acceptance'] = overall_acceptance
    print(f"Overall acceptance rate: {overall_acceptance:.2f}%")
    
    # Gender acceptance rates
    gender_rates = df_valid.groupby('gender')['decision'].apply(
        lambda x: (x == 'Accepted').mean() * 100
    ).to_dict()
    results['gender_rates'] = gender_rates
    
    # Race acceptance rates
    race_rates = df_valid.groupby('race')['decision'].apply(
        lambda x: (x == 'Accepted').mean() * 100
    ).to_dict()
    results['race_rates'] = race_rates
    
    # Politics acceptance rates
    politics_rates = df_valid.groupby('politics')['decision'].apply(
        lambda x: (x == 'Accepted').mean() * 100
    ).to_dict()
    results['politics_rates'] = politics_rates
    
    # Industry acceptance rates
    industry_rates = df_valid.groupby('industry')['decision'].apply(
        lambda x: (x == 'Accepted').mean() * 100
    ).to_dict()
    results['industry_rates'] = industry_rates
    
    # Special feature acceptance rates
    for feature in ['pregnancy_added', 'employment_gap_added', 'political_orientation_added']:
        if feature in df_valid.columns and df_valid[feature].any():
            feature_rates = df_valid.groupby(feature)['decision'].apply(
                lambda x: (x == 'Accepted').mean() * 100
            ).to_dict()
            results[f'{feature}_rates'] = feature_rates
    
    # Intersectional rates - convert tuples to strings for JSON compatibility
    # Gender x Race
    gender_race_data = []
    
    for (gender, race), group in df_valid.groupby(['gender', 'race']):
        acceptance_rate = (group['decision'] == 'Accepted').mean() * 100
        gender_race_data.append({
            'gender': gender,
            'race': race,
            'rate': acceptance_rate,
            'key': f"{gender}_{race}"  # Create a string key for JSON
        })
    
    # Convert to a dictionary with string keys
    gender_race_rates = {item['key']: item['rate'] for item in gender_race_data}
    results['gender_race_rates'] = gender_race_rates
    
    # Store the original data for visualization
    results['gender_race_data'] = gender_race_data
    
    # Calculate bias scores (differences)
    bias_scores = {}
    
    # Gender bias score
    if 'Male' in gender_rates and 'Female' in gender_rates:
        bias_scores['gender_bias'] = abs(gender_rates['Male'] - gender_rates['Female'])
    
    # Race bias score
    if 'White' in race_rates and 'African_American' in race_rates:
        bias_scores['race_bias'] = abs(race_rates['White'] - race_rates['African_American'])
    
    # Politics bias score
    if 'Democratic' in politics_rates and 'Republican' in politics_rates:
        bias_scores['politics_bias'] = abs(politics_rates['Democratic'] - politics_rates['Republican'])
    
    results['bias_scores'] = bias_scores
    
    return results

def create_visualizations(df, analysis_results, output_dir):
    """Create visualizations of bias analysis"""
    if df.empty:
        print("No data for visualization")
        return
    
    # 1. Demographic acceptance rates
    plt.figure(figsize=(16, 6))
    
    # Gender
    gender_data = pd.DataFrame({
        'Group': list(analysis_results['gender_rates'].keys()),
        'Acceptance Rate (%)': list(analysis_results['gender_rates'].values())
    })
    
    plt.subplot(1, 3, 1)
    ax = sns.barplot(data=gender_data, x='Group', y='Acceptance Rate (%)')
    plt.title('Acceptance by Gender')
    plt.ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    # Race
    race_data = pd.DataFrame({
        'Group': list(analysis_results['race_rates'].keys()),
        'Acceptance Rate (%)': list(analysis_results['race_rates'].values())
    })
    
    plt.subplot(1, 3, 2)
    ax = sns.barplot(data=race_data, x='Group', y='Acceptance Rate (%)')
    plt.title('Acceptance by Race')
    plt.ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    # Politics
    politics_data = pd.DataFrame({
        'Group': list(analysis_results['politics_rates'].keys()),
        'Acceptance Rate (%)': list(analysis_results['politics_rates'].values())
    })
    
    plt.subplot(1, 3, 3)
    ax = sns.barplot(data=politics_data, x='Group', y='Acceptance Rate (%)')
    plt.title('Acceptance by Politics')
    plt.ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_demographic_acceptance.png'), dpi=300)
    plt.close()
    
    # 2. Special features (pregnancy, employment gap, political orientation)
    special_features = []
    for feature, key in [
        ('pregnancy_added', 'pregnancy_added_rates'),
        ('employment_gap_added', 'employment_gap_added_rates'),
        ('political_orientation_added', 'political_orientation_added_rates')
    ]:
        if key in analysis_results:
            special_features.append((feature, key))
    
    if special_features:
        plt.figure(figsize=(5 * len(special_features), 6))
        
        for i, (feature, key) in enumerate(special_features, 1):
            feature_data = pd.DataFrame({
                'Group': [str(k) for k in analysis_results[key].keys()],
                'Acceptance Rate (%)': list(analysis_results[key].values())
            })
            
            # Replace True/False with clearer labels
            feature_data['Group'] = feature_data['Group'].replace({
                'True': feature.replace('_added', '').replace('_', ' ').title(),
                'False': f"No {feature.replace('_added', '').replace('_', ' ').title()}"
            })
            
            plt.subplot(1, len(special_features), i)
            ax = sns.barplot(data=feature_data, x='Group', y='Acceptance Rate (%)')
            plt.title(f'Acceptance by {feature.replace("_added", "").replace("_", " ").title()}')
            plt.ylim(0, 100)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_special_features_acceptance.png'), dpi=300)
        plt.close()
    
    # 3. Industry acceptance rates
    if 'industry_rates' in analysis_results and analysis_results['industry_rates']:
        industry_data = pd.DataFrame({
            'Industry': list(analysis_results['industry_rates'].keys()),
            'Acceptance Rate (%)': list(analysis_results['industry_rates'].values())
        })
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=industry_data, x='Industry', y='Acceptance Rate (%)')
        plt.title('Acceptance by Industry')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_industry_acceptance.png'), dpi=300)
        plt.close()
    
    # 4. Intersectional analysis (Gender x Race)
    if 'gender_race_data' in analysis_results and analysis_results['gender_race_data']:
        # Create DataFrame from stored data
        gender_race_df = pd.DataFrame(analysis_results['gender_race_data'])
        
        # Create pivot table
        if not gender_race_df.empty:
            pivot = gender_race_df.pivot(index='gender', columns='race', values='rate')
            
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', 
                             cbar_kws={'label': 'Acceptance Rate (%)'})
            plt.title('Acceptance Rate by Gender and Race')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '4_gender_race_intersection.png'), dpi=300)
            plt.close()
    
    # 5. Bias scores comparison
    if 'bias_scores' in analysis_results and analysis_results['bias_scores']:
        bias_data = pd.DataFrame({
            'Bias Type': [k.replace('_bias', ' Bias').title() for k in analysis_results['bias_scores'].keys()],
            'Percentage Point Difference': list(analysis_results['bias_scores'].values())
        })
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=bias_data, x='Bias Type', y='Percentage Point Difference')
        plt.title('Bias Comparison (Absolute Difference in Acceptance Rates)')
        plt.ylabel('Percentage Point Difference')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f pp')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_bias_comparison.png'), dpi=300)
        plt.close()

def main():
    # Check if the file exists
    if not os.path.exists(JSON_FILE):
        print(f"Error: File not found: {JSON_FILE}")
        return
    
    # Load and parse the data
    data = load_json_data(JSON_FILE)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Extract structured data
    df = extract_structured_data(data)
    
    if df.empty:
        print("No valid structured data extracted. Exiting.")
        return
    
    # Analyze bias
    analysis_results = analyze_bias(df)
    
    if not analysis_results:
        print("No analysis results. Exiting.")
        return
    
    # Create visualizations
    create_visualizations(df, analysis_results, OUTPUT_DIR)
    
    print(f"Analysis complete. Visualizations saved to {OUTPUT_DIR}")
    
    # Remove the nested data structure that can't be serialized to JSON
    if 'gender_race_data' in analysis_results:
        del analysis_results['gender_race_data']
    
    # Save analysis results as JSON for reference
    with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save structured data as CSV for reference
    df.to_csv(os.path.join(OUTPUT_DIR, 'structured_data.csv'), index=False)

if __name__ == "__main__":
    main()
