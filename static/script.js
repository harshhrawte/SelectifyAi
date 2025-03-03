// Image Slider Logic
let currentSlide = 0;

document.addEventListener("DOMContentLoaded", function () {
    const slides = document.querySelectorAll('.slider-image');

    function showSlide(index) {
        slides.forEach((slide, i) => {
            slide.classList.remove('active', 'animate');
            if (i === index) {
                slide.classList.add('active');
                setTimeout(() => slide.classList.add('animate'), 50);
            }
        });
    }

    function changeSlide(direction) {
        currentSlide = (currentSlide + direction + slides.length) % slides.length;
        showSlide(currentSlide);
    }

    showSlide(currentSlide);
    setInterval(() => changeSlide(1), 5000);
});

// File Upload Handling
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const proceedButton = document.getElementById('proceed-button');

let uploadedFiles = [];

fileInput.addEventListener('change', function () {
    proceedButton.style.display = 'none';
    
    for (let file of fileInput.files) {
        if (uploadedFiles.some(f => f.name === file.name)) {
            alert(`File ${file.name} is already uploaded.`);
            continue;
        }

        const fileExtension = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'docx'].includes(fileExtension)) {
            alert('Only PDF and DOCX files are allowed.');
            return;
        }

        uploadedFiles.push(file);

        const fileElement = document.createElement('div');
        fileElement.classList.add('file-item');

        const fileName = document.createElement('span');
        fileName.textContent = file.name;
        fileElement.appendChild(fileName);

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.classList.add('delete-btn');
        deleteBtn.onclick = () => {
            uploadedFiles = uploadedFiles.filter(f => f.name !== file.name);
            fileElement.remove();
            if (uploadedFiles.length === 0) proceedButton.style.display = 'none';
        };

        fileElement.appendChild(deleteBtn);
        fileList.appendChild(fileElement);
    }

    if (uploadedFiles.length > 0) {
        proceedButton.style.display = 'block';
    }
});

// Job Description & Skills Handling
const jobDescInput = document.getElementById('jobDescription');
const skillsInput = document.getElementById('skillInput');
const skillsList = document.getElementById('skillsList');

function addSkill(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        const skill = skillsInput.value.trim();
        if (skill) {
            const skillItem = document.createElement('li');
            skillItem.textContent = skill;

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'X';
            deleteBtn.classList.add('delete-btn');
            deleteBtn.onclick = () => skillItem.remove();

            skillItem.appendChild(deleteBtn);
            skillsList.appendChild(skillItem);
            skillsInput.value = '';
        }
    }
}

skillsInput.addEventListener('keypress', addSkill);

// Handle Form Submission
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const jobDesc = jobDescInput.value.trim();
    const skills = Array.from(skillsList.children).map(li => li.firstChild.textContent.trim());

    if (!jobDesc) {
        alert("Please enter a job description.");
        return;
    }
    if (skills.length === 0) {
        alert("Please enter at least one skill.");
        return;
    }
    if (uploadedFiles.length === 0) {
        alert("Please upload at least one resume.");
        return;
    }

    const formData = new FormData();
    formData.append("job_description", jobDesc);
    formData.append("skills", skills.join(','));

    for (const file of uploadedFiles) {
        formData.append("resumes", file);
    }

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        if (response.ok) {
            console.log("Upload successful, redirecting...");
            window.location.href = "/ranked-resume"; // Update here
        } else {
            alert("Error processing the resumes. Try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while uploading. Please try again.");
    }

});

// Fetch Ranked Resumes on ranked_resume.html Load
document.addEventListener("DOMContentLoaded", async function () {
    if (window.location.pathname.includes('ranked_resume.html')) {
        try {
            const response = await fetch('/get-ranked-resumes');
            if (!response.ok) {
                throw new Error(`Failed to fetch rankings. Status: ${response.status}`);
            }

            const rankedResumes = await response.json();
            if (!Array.isArray(rankedResumes) || rankedResumes.length === 0) {
                alert("No resumes ranked yet. Please wait and try again.");
                return;
            }

            const resumeList = document.getElementById('resume-list');
            resumeList.innerHTML = "";

            rankedResumes.forEach((resume, index) => {
                const name = resume?.name || "Unknown";
                const score = isNaN(resume?.final_score) ? "N/A" : resume.final_score.toFixed(2);

                const listItem = document.createElement('li');
                listItem.textContent = `${index + 1}. ${name} (Score: ${score})`;
                resumeList.appendChild(listItem);
            });

            document.getElementById('ranked-resumes').style.display = 'block';
        } catch (error) {
            console.error("Fetch error:", error);
        }
    }
});
