import boto3
import utils
import json
import pandas as pd
import plotly.express as px
import traceback
import chat

from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate

import logging
import sys
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("cost_analysis")

def get_cost_analysis(days: str=30):
    """Cost analysis data collection"""
    logger.info(f"Getting cost analysis...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client('ce')

        # service cost
        service_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        # logger.info(f"Service Cost: {service_response}")
        
        service_costs = pd.DataFrame([
            {
                'SERVICE': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in service_response['ResultsByTime'][0]['Groups']
        ])
        logger.info(f"Service Costs: {service_costs}")
        
        # region cost
        region_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'REGION'}]
        )
        # logger.info(f"Region Cost: {region_response}")
        
        region_costs = pd.DataFrame([
            {
                'REGION': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in region_response['ResultsByTime'][0]['Groups']
        ])
        logger.info(f"Region Costs: {region_costs}")
        
        # Daily Cost
        daily_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        # logger.info(f"Daily Cost: {daily_response}")
        
        daily_costs = []
        for time_period in daily_response['ResultsByTime']:
            date = time_period['TimePeriod']['Start']
            for group in time_period['Groups']:
                daily_costs.append({
                    'date': date,
                    'SERVICE': group['Keys'][0],
                    'cost': float(group['Metrics']['UnblendedCost']['Amount'])
                })
        
        daily_costs_df = pd.DataFrame(daily_costs)
        logger.info(f"Daily Costs: {daily_costs_df}")
        
        return {
            'service_costs': service_costs,
            'region_costs': region_costs,
            'daily_costs': daily_costs_df
        }
        
    except Exception as e:
        logger.info(f"Error in cost analysis: {str(e)}")
        return None

def create_cost_visualizations(cost_data):
    """Cost Visualization"""
    logger.info("Creating cost visualizations...")

    if not cost_data:
        logger.info("No cost data available")
        return None
        
    visualizations = {}
    
    # service cost (pie chart)
    fig_pie = px.pie(
        cost_data['service_costs'],
        values='cost',
        names='SERVICE',
        title='Service Cost'
    )
    visualizations['service_pie'] = fig_pie
            
    # daily trend cost (line chart)
    fig_line = px.line(
        cost_data['daily_costs'],
        x='date',
        y='cost',
        color='SERVICE',
        title='Daily Cost Trend'
    )
    visualizations['daily_trend'] = fig_line
    
    # region cost (bar chart)
    fig_bar = px.bar(
        cost_data['region_costs'],
        x='REGION',
        y='cost',
        title='Region Cost'
    )
    visualizations['region_bar'] = fig_bar
    
    logger.info(f"Visualizations created: {list(visualizations.keys())}")
    return visualizations

def generate_cost_insights():
    if cost_data:
        cost_data_dict = {
            'service_costs': cost_data['service_costs'].to_dict(orient='records'),
            'region_costs': cost_data['region_costs'].to_dict(orient='records'),
            'daily_costs': cost_data['daily_costs'].to_dict(orient='records') if 'daily_costs' in cost_data else []
        }
    else:
        return "Not available"

    system = (
        "You are an AWS solutions architect."
        "Use the following Cost Data to answer the user's questions."
        "If you don't know the answer, honestly say you don't know."
        "Explain the reason for your answer clearly and thoroughly."
    )
    human = (
        "Analyze the following AWS cost data and provide detailed insights:"
        "Cost Data:"
        "{raw_cost}"
        
        "Please analyze the following items:"
        "1. Major cost drivers"
        "2. Abnormal patterns or sudden cost increases"
        "3. Areas where cost optimization is possible"
        "4. Overall cost trends and future predictions"
        
        "Please provide the analysis results in the following format:"

        "### Major Cost Drivers"
        "- [Specific analysis content]"

        "### Anomaly Pattern Analysis"
        "- [Explanation of abnormal cost patterns]"

        "### Optimization Opportunities"
        "- [Specific optimization strategies]"

        "### Cost Trends"
        "- [Trend analysis and predictions]"
    ) 

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # logger.info('prompt: ', prompt)    

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm

    raw_cost = json.dumps(cost_data_dict)

    try:
        response = chain.invoke(
            {
                "raw_cost": raw_cost
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return response.content

cost_data = {}
visualizations = {}
insights = ""

def get_visualiation():
    global cost_data, visualizations

    try:
        cost_data = get_cost_analysis()
        if cost_data:
            logger.info(f"No cost data available")

            # draw visualizations        
            visualizations = create_cost_visualizations(cost_data)

    except Exception as e:
        logger.info(f"Error to earn cost data: {str(e)}")   

get_visualiation() 

def ask_cost_insights(question):
    if cost_data:
        cost_data_dict = {
            'service_costs': cost_data['service_costs'].to_dict(orient='records'),
            'region_costs': cost_data['region_costs'].to_dict(orient='records'),
            'daily_costs': cost_data['daily_costs'].to_dict(orient='records') if 'daily_costs' in cost_data else []
        }
    else:
        return "Cost 데이터를 가져오는데 실패하였습니다."

    system = (
        "You are an AWS solutions architect."
        "Use the following Cost Data to answer the user's questions."
        "If you don't know the answer, honestly say you don't know."
        "Explain the reason for your answer clearly and thoroughly."
    )
    human = (
        "Question: {question}"

        "Cost Data:"
        "{raw_cost}"        
    ) 

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # logger.info('prompt: ', prompt)    

    llm = chat.get_chat()
    chain = prompt | llm

    raw_cost = json.dumps(cost_data_dict)

    try:
        response = chain.invoke(
            {
                "question": question,
                "raw_cost": raw_cost
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return response.content
