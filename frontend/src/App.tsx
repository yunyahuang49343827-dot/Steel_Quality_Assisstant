import {
  Bot,
  CircleHelp,
  Factory,
  FlaskConical,
  Layers3,
} from "lucide-react";

import {
  useEffect,
  useState,
} from "react";

import OverviewPage
  from "./pages/OverviewPage";

import PredictionLab
  from "./components/PredictionLab";

import AICopilot
  from "./components/AICopilot";


type SectionId =
  | "overview"
  | "defect-intelligence"
  | "prediction-lab"
  | "ai-copilot";


function App() {

  const [
    activeSection,
    setActiveSection,
  ] = useState<SectionId>(
    "overview"
  );


  function scrollToSection(
    id: SectionId
  ) {

    if (id === "overview") {

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });

      return;
    }


    document
      .getElementById(id)
      ?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
  }


  useEffect(
    () => {

      const handleScroll = () => {

        const sectionIds: SectionId[] = [
          "overview",
          "defect-intelligence",
          "prediction-lab",
          "ai-copilot",
        ];


        const triggerPoint =
          window.scrollY + 140;


        let currentSection: SectionId =
          "overview";


        for (const id of sectionIds) {

          if (id === "overview") {
            continue;
          }


          const element =
            document.getElementById(id);


          if (
            element &&
            element.offsetTop <= triggerPoint
          ) {
            currentSection = id;
          }
        }


        setActiveSection(
          currentSection
        );
      };


      handleScroll();


      window.addEventListener(
        "scroll",
        handleScroll,
        {
          passive: true,
        }
      );


      return () => {

        window.removeEventListener(
          "scroll",
          handleScroll
        );
      };

    },
    []
  );


  return (
    <div className="app-shell">

      <header className="top-header">

        <div className="brand">

          <div className="brand-icon">
            <Layers3 size={25} />
          </div>

          <div>

            <h1>
              Steel Quality Intelligence
            </h1>

            <p>
              品質分析 · ML 預測 ·
              可解釋性 · AI Copilot
            </p>

          </div>

        </div>


        <nav className="main-nav">

          <button
            className={
              activeSection === "overview"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={
              () =>
                scrollToSection(
                  "overview"
                )
            }
          >
            品質總覽
          </button>


          <button
            className={
              activeSection === "defect-intelligence"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={
              () =>
                scrollToSection(
                  "defect-intelligence"
                )
            }
          >
            缺陷分析
          </button>


          <button
            className={
              activeSection === "prediction-lab"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={
              () =>
                scrollToSection(
                  "prediction-lab"
                )
            }
          >
            Prediction Lab
          </button>


          <button
            className={
              activeSection === "ai-copilot"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={
              () =>
                scrollToSection(
                  "ai-copilot"
                )
            }
          >
            AI Copilot
          </button>

        </nav>


        <div className="header-actions">

          <button
            className="icon-button"
            aria-label="說明"
          >
            <CircleHelp size={19} />
          </button>

          <div className="header-divider" />

          <div className="system-pill">

            <span className="status-dot" />

            API 已連線

          </div>

        </div>

      </header>


      <main className="page-container">

        <section
          id="overview"
          className="overview-anchor"
        >

          <div className="page-intro">

            <div>

              <span className="page-eyebrow">

                <Factory size={14} />

                製造品質分析

              </span>

              <h2>
                品質總覽
              </h2>

              <p>
                從同一個工作區掌握缺陷組成、
                模型表現與預測證據。
              </p>

            </div>


            <div className="environment-pill">

              <FlaskConical size={15} />

              Portfolio Demo Environment

            </div>

          </div>


          <OverviewPage />

        </section>


        <PredictionLab />

        <AICopilot />

      </main>


      <button
        className="copilot-floating"
        title="AI Quality Copilot"
        onClick={
          () =>
            scrollToSection(
              "ai-copilot"
            )
        }
      >

        <Bot size={22} />

      </button>

    </div>
  );
}


export default App;