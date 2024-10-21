using UnityEngine.UIElements;
using UnityEngine.SceneManagement;
using UnityEngine;

namespace PrototypeUI
{
    /// <summary>
    /// Simple button that loads the scene
    /// </summary>
    [UxmlElement]
    public partial class GoToSceneButton : Button
    {
        [UxmlAttribute]
        public string SceneName { get; set; }

        public static readonly new string ussClassName = "prototype-button";

        public GoToSceneButton() : base()
        {
            AddToClassList(ussClassName);
            clicked += GoToScene;
        }

        private void GoToScene()
        {
            Debug.Log($"Loading scene {SceneName}");
            SceneManager.LoadScene(SceneName);
        }
    }
}
