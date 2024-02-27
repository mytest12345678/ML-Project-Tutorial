# Tutorial Guide:

## ML-Project-Tutorial: 
A simple documented example of how to create an ML Project Workflow and how to manage a team to develop it over time. 

### ML Work Flow
1. ML Project Scoping
  - Identify and Document the Scope of the Project's Problem Space and it's qualities
  - Identify and Document the Scope of the Project's Solution Space and it's qualities
2. Project Env Prep
  - Install if needed and Import libs
  - Remove or Archive any previous Training Session's Files before each New Training Session
3. Downloading/Querrying and Loading Data
4. Data Exploration 
  - Aggregate Stat Summarries
  - Visualizations
  - Update Project Scope
5. Training and Hyperparameter Tuning:
  - Preprocessing Settings Selection
  - Model and Model Settings Selection
  - Training Settings Selection
  - Monitoring and Visualizations
7. Data drift and A/B Testing for QA
  - test all subsets of val or test set data that you are concerned about the average quality of the end user experience. 
8. Model Serving and Production Monitoring
  - MLOps
  - CICD
  - visualizations
  - email alerts

### Initial ML Project Development Setup Steps: See file ML_Project_POC_Tutorial.ipynb
1. Hard code a bare-bones proof-of-concept (POC) for an ML Project, within the identified project scope, in a single jupyter notebook file.
2. Go back and Define functions for repeatable lines of code and define variables for controling previously hardcoded settings while improving naming conventions.
3. Move functions into a seperate file and import them into the main file to make it simpler.
4. Add Documentation and make a pull request
### Development Steps for AGILE Sprints:
1. Select an issue in the project to work on from the current Sprint and a assign it to yourself to get email updates.
2. Add or debug the documented feature  Features documented in the repo's issues over time. 

### Project Management Steps: ML Project Tutorial
https://github.com/orgs/mytest12345678/projects/2
1. Setting up a Git Hub Project Dashboard and Repo.
  - Set Rules for the repo to protect the main branch by requiring reviews for push requests and prevent direct commits
  - Document Standards and naming conventions for repo branching if not yet established 
  - Add properties to attatch to issues such as priority, size, sprint, labels, tags, etc.
  - Add meaningful views to the dashboard to help orginize the project's issues by their properties.
  - Document standards for issue assignments and team members posting to the project dashboard in the description.
2. Attend regularly sceduled meetings with your team and with other managers from other departments.
3. Manually assign issues for team members to work on or allow self assigning issues for a more hands free approach to managing the team.
  - manually assigning issues is only advisable for small teams
4. Allways encourage team members to ask questions for clarity on the goal of a tasks and encourage them to ask for more directions if they get stuck.
  - Identify how they view the problem, how they are trying to sove it and how they arrived at these conclusions.
  - Identify what's the problem with this approach for solving the task
  - Identify a proper approach to solving the task
  - Identify what steps they could taken to better understand the problem and eventually come up with a similar viable solution to the problem by seaking out the aswers to a set of well framed questions online.
  - Then, using all that information, ask questions to lead them towards the answer without just giving them the solution.
  - Time spent mentoring juniors is an investment into their potential and never a waste of time if done properly.


