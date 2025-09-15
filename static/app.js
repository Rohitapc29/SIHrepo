// static/js/app.js

// Flash messages auto-dismiss after 4s
document.addEventListener("DOMContentLoaded", () => {
  const flashes = document.querySelectorAll(".flash");
  flashes.forEach(flash => {
    setTimeout(() => {
      flash.style.opacity = "0";
      setTimeout(() => flash.remove(), 600); // remove after fade
    }, 4000);
  });

  // Simple form helper: prevent empty submits
  const forms = document.querySelectorAll("form");
  forms.forEach(form => {
    form.addEventListener("submit", (e) => {
      let valid = true;
      form.querySelectorAll("input[required], textarea[required]").forEach(field => {
        if (!field.value.trim()) {
          field.classList.add("error");
          valid = false;
        } else {
          field.classList.remove("error");
        }
      });
      if (!valid) {
        e.preventDefault();
        alert("Please fill out all required fields!");
      }
    });
  });
});
