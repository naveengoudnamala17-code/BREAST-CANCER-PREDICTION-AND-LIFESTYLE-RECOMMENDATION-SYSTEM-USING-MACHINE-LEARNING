document.addEventListener("DOMContentLoaded", function () {

    const customizeBtn = document.getElementById("customizeBtn");
    const customPanel = document.getElementById("customPanel");
    const closeCustom = document.getElementById("closeCustom");

    const bgColor = document.getElementById("bgColor");
    const cardColor = document.getElementById("cardColor");
    const textColor = document.getElementById("textColor");
    const dashboardTitle = document.getElementById("dashboardTitle");

    const applyTheme = document.getElementById("applyTheme");
    const resetTheme = document.getElementById("resetTheme");


    /* OPEN PANEL */

    customizeBtn.addEventListener("click", function () {
        customPanel.classList.add("active");
    });


    /* CLOSE PANEL */

    closeCustom.addEventListener("click", function () {
        customPanel.classList.remove("active");
    });


    /* APPLY THEME */

    applyTheme.addEventListener("click", function () {

        const background = bgColor.value;
        const card = cardColor.value;
        const text = textColor.value;

        /* Background */

        document.body.style.backgroundColor = background;


        /* Cards */

        const cards = document.querySelectorAll(
            ".card, .stat-card, .dashboard-card"
        );

        cards.forEach(function (item) {
            item.style.backgroundColor = card;
        });


        /* Text */

        document.body.style.color = text;


        /* Dashboard title */

        const title = document.querySelector(
            ".dashboard-title, .page-title, h1"
        );

        if (title && dashboardTitle.value.trim() !== "") {
            title.textContent = dashboardTitle.value;
        }


        /* Save settings */

        localStorage.setItem("dashboard_bg", background);
        localStorage.setItem("dashboard_card", card);
        localStorage.setItem("dashboard_text", text);
        localStorage.setItem(
            "dashboard_title",
            dashboardTitle.value
        );

        alert("Dashboard theme updated successfully!");
    });


    /* RESET */

    resetTheme.addEventListener("click", function () {

        const defaultBg = "#050b14";
        const defaultCard = "#0d6efd";
        const defaultText = "#ffffff";
        const defaultTitle = "Breast Cancer AI";


        bgColor.value = defaultBg;
        cardColor.value = defaultCard;
        textColor.value = defaultText;
        dashboardTitle.value = defaultTitle;


        document.body.style.backgroundColor = defaultBg;
        document.body.style.color = defaultText;


        const cards = document.querySelectorAll(
            ".card, .stat-card, .dashboard-card"
        );

        cards.forEach(function (item) {
            item.style.backgroundColor = defaultCard;
        });


        const title = document.querySelector(
            ".dashboard-title, .page-title, h1"
        );

        if (title) {
            title.textContent = defaultTitle;
        }


        localStorage.removeItem("dashboard_bg");
        localStorage.removeItem("dashboard_card");
        localStorage.removeItem("dashboard_text");
        localStorage.removeItem("dashboard_title");

    });


    /* LOAD SAVED THEME */

    const savedBg = localStorage.getItem("dashboard_bg");
    const savedCard = localStorage.getItem("dashboard_card");
    const savedText = localStorage.getItem("dashboard_text");
    const savedTitle = localStorage.getItem("dashboard_title");


    if (savedBg) {
        bgColor.value = savedBg;
        document.body.style.backgroundColor = savedBg;
    }

    if (savedCard) {
        cardColor.value = savedCard;

        const cards = document.querySelectorAll(
            ".card, .stat-card, .dashboard-card"
        );

        cards.forEach(function (item) {
            item.style.backgroundColor = savedCard;
        });
    }

    if (savedText) {
        textColor.value = savedText;
        document.body.style.color = savedText;
    }

    if (savedTitle) {
        dashboardTitle.value = savedTitle;

        const title = document.querySelector(
            ".dashboard-title, .page-title, h1"
        );

        if (title) {
            title.textContent = savedTitle;
        }
    }

});