// install axios first!: yarn add axios
// based on https://www.meziantou.net/deleting-github-actions-artifacts-using-the-github-rest-api.htm
const axios = require('axios');

// https://github.com/settings/tokens/new
const githubToken = 'ghp_xxxxxxxxxxxxxxxxxx'

const httpClient = axios.create({
    baseURL: 'https://api.github.com',
    headers: {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': `token ${githubToken}`,
        'User-Agent': 'ArtifactsCleaner/1.0'
    }
});

async function getAllProjects() {
    try {
        let page = 1;
        const pageSize = 100;
        let allProjects = [];
        let response;

        do {
            response = await httpClient.get(`/user/repos?per_page=${pageSize}&page=${page}`);
            allProjects = allProjects.concat(response.data.map(repo => repo.full_name));
            page++;
        } while (response.data.length === pageSize);

        return allProjects;
    } catch (err) {
        console.error('Error fetching repositories:', err);
        return [];
    }
}

async function deleteOldArtifacts(projects) {
    console.log(`Exploring ${projects.length} repos...`);

    for (const project of projects) {
        let pageIndex = 1;
        const pageSize = 100;
        let page;

        do {
            const url = `/repos/${project}/actions/artifacts?per_page=${pageSize}&page=${pageIndex}`;
            try {
                const response = await httpClient.get(url);
                page = response.data;
                for (const item of page.artifacts) {
                    if (!item.expired && new Date(item.created_at) < new Date(Date.now() - 24 * 60 * 60 * 1000)) {
                        const deleteUrl = `/repos/${project}/actions/artifacts/${item.id}`;
                        try {
                            await httpClient.delete(deleteUrl);
                            console.log(`Deleted: ${item.name} created at ${item.created_at} from ${project}`);
                        } catch (deleteErr) {
                            console.error(`Error ${deleteErr} deleting artifact: ${item.name} in ${project}, ${deleteErr}`);
                        }
                    }
                }
                pageIndex++;
            } catch (err) {
                console.error(`Error retrieving artifacts for ${project}: ${err}`);
                break;
            }
        } while (page.artifacts.length >= pageSize);
    }
}

async function cleanUpArtifacts() {
    const projects = await getAllProjects();
    await deleteOldArtifacts(projects);
}

cleanUpArtifacts();
